import os
import argparse
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from tensorboardX import SummaryWriter
from imgaug import augmenters as iaa
from misc.train_ultils_all_iter import *
import math
from loss.mtmr_loss import get_loss_mtmr
from loss.rank_ordinal_loss import cost_fn
from loss.dorn_loss import OrdinalLoss
import dataset as dataset
from config import Config
from loss.ceo_loss import CEOLoss, FocalLoss, SoftLabelOrdinalLoss, FocalOrdinalLoss, count_pred
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.distributions import normal
from models import SCUBaNet

####
class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class Trainer(Config):
    def __init__(self, _args=None):
        super(Trainer, self).__init__(_args=_args)
        if _args is not None:
            self.__dict__.update(_args.__dict__)
            print(self.run_info)

        self.exp_args = _args

    ####
    def view_dataset(self, mode='train'):
        train_pairs, valid_pairs = getattr(dataset, ('prepare_%s_data' % self.dataset))()
        if mode == 'train':
            train_augmentors = self.train_augmentors()
            ds = dataset.DatasetSerial(train_pairs, has_aux=False,
                                       shape_augs=iaa.Sequential(train_augmentors[0]),
                                       input_augs=iaa.Sequential(train_augmentors[1]))
        else:
            infer_augmentors = self.infer_augmentors()  # HACK
            ds = dataset.DatasetSerial(valid_pairs, has_aux=False,
                                       shape_augs=iaa.Sequential(infer_augmentors)[0])
        dataset.visualize(ds, 4)
        return

    ####
    def train_step(self, engine, net, batch, iters, scheduler, optimizer, device):
        net.train()  # train mode

        imgs_cpu, graphs_cpu, adjs_cpu, true_cpu = batch
        # imgs_cpu = imgs_cpu.permute(0, 3, 1, 2)  # to NCHW
        scheduler.step(engine.state.epoch + engine.state.iteration / iters)  # scheduler.step(epoch + i / iters)
        # push data to GPUs
        imgs = imgs_cpu.to(device).float()
        true = true_cpu.to(device).long()  # not one-hot

        if isinstance(graphs_cpu, list):
            graphs = [graphs_cpu[0].to(device).float(), graphs_cpu[1].to(device).float()]
            adjs = [adjs_cpu[0].to(device).float(), adjs_cpu[1].to(device).float()]
        else:
            graphs = graphs_cpu.to(device).float()
            adjs = adjs_cpu.to(device).float()
        # -----------------------------------------------------------
        net.zero_grad()  # not rnn so not accumulate
        out_net = net(imgs, graphs, adjs)  # a list contains all the out put of the network
        loss = 0.

        # assign output
        if "CLASS" in self.task_type:
            logit_class = out_net
        if "REGRESS" in self.task_type:
            if ("rank_ordinal" in self.loss_type) or ("dorn" in self.loss_type):
                logit_regress, probas = out_net[0], out_net[1]
            else:
                logit_regress = out_net
        if "MULTI" in self.task_type:
            logit_class, logit_regress = out_net[0], out_net[1]

        # compute loss function
        if "ce" in self.loss_type:
            if isinstance(logit_class, list):
                logits_class = logit_class
                for logit_class in logits_class:
                    prob = F.softmax(logit_class, dim=-1)
                    loss_entropy = F.cross_entropy(logit_class, true, reduction='mean')
                    pred = torch.argmax(prob, dim=-1)
                    loss += loss_entropy
            else:
                prob = F.softmax(logit_class, dim=-1)
                loss_entropy = F.cross_entropy(logit_class, true, reduction='mean')
                pred = torch.argmax(prob, dim=-1)
                loss += loss_entropy

        if 'FocalLoss' in self.loss_type:
            loss_focal = FocalLoss()(logit_class, true)
            prob = F.softmax(logit_class, dim=-1)
            pred = torch.argmax(prob, dim=-1)
            loss += loss_focal

        if "mse" in self.loss_type:
            criterion = torch.nn.MSELoss()
            loss_regres = criterion(logit_regress, true.float())
            loss += loss_regres
            if "REGRESS" in self.task_type:
                label = torch.tensor([0., 1., 2., 3.]).repeat(len(true), 1).permute(1, 0).cuda()
                pred = torch.argmin(torch.abs(logit_regress - label), 0)
        
        if "mae" in self.loss_type:
            criterion = torch.nn.L1Loss()
            loss_regres = criterion(logit_regress, true.float())
            loss += loss_regres
            if "REGRESS" in self.task_type:
                label = torch.tensor([0., 1., 2., 3.]).repeat(len(true), 1).permute(1, 0).cuda()
                pred = torch.argmin(torch.abs(logit_regress - label), 0)
        
        if "soft_label" in self.loss_type:
            criterion = SoftLabelOrdinalLoss(alpha=self.alpha)
            loss_regres = criterion(logit_regress, true.float())
            loss += loss_regres
            if "REGRESS" in self.task_type:
                label = torch.tensor([0., 1 / 3, 2 / 3, 1.]).repeat(len(true), 1).permute(1, 0).cuda()
                pred = torch.argmin(torch.abs(logit_regress - label), 0)
        
        if "FocalOrdinal" in self.loss_type:
            criterion = FocalOrdinalLoss(pooling=True)
            loss_regres = criterion(logit_regress, true.float())
            loss += loss_regres
            pred = count_pred(logit_regress)
        if "ceo" in self.loss_type:
            criterion = CEOLoss(num_classes=self.nr_classes)
            loss_ordinal = criterion(logit_regress, true)
            loss += loss_ordinal
        
        if "mtmr" in self.loss_type:
            loss = get_loss_mtmr(logit_class, logit_regress, true, true)
            prob = F.softmax(logit_class, dim=-1)
            pred = torch.argmax(prob, dim=-1)
        
        if "rank_coral" in self.loss_type:
            loss = cost_fn(logit_regress, true)
            predict_levels = probas > 0.5
            pred = torch.sum(predict_levels, dim=1)
        
        if "rank_dorn" in self.loss_type:
            pred, softmax = net(imgs)  # forward
            loss = OrdinalLoss()(softmax, true)

        acc = torch.mean((pred == true).float())  # batch accuracy
        # gradient update
        loss.backward()
        optimizer.step()

        # -----------------------------------------------------------
        return dict(
            loss=loss.item(),
            acc=acc.item(),
        )

    ####
    def infer_step(self, net, batch, device):
        net.eval()  # infer mode

        imgs_cpu, graphs_cpu, adjs_cpu, true_cpu = batch
        # imgs_cpu = imgs_cpu.permute(0, 3, 1, 2)  # to NCHW
        # push data to GPUs
        
        imgs = imgs_cpu.to(device).float()
        true = true_cpu.to(device).long()  # not one-hot

        if isinstance(graphs_cpu, list):
            graphs = [graphs_cpu[0].to(device).float(), graphs_cpu[1].to(device).float()]
            adjs = [adjs_cpu[0].to(device).float(), adjs_cpu[1].to(device).float()]
        else:
            graphs = graphs_cpu.to(device).float()
            adjs = adjs_cpu.to(device).float()

        # -----------------------------------------------------------
        with torch.no_grad():  # dont compute gradient
            out_net = net(imgs, graphs, adjs)  # a list contains all the out put of the network
            if "CLASS" in self.task_type:
                logit_class = out_net
                if isinstance(logit_class, list):
                    logit_class = logit_class[-1]
                prob = nn.functional.softmax(logit_class, dim=-1)
                return dict(logit_c=prob.cpu().numpy(),  # from now prob of class task is called by logit_c
                            true=true.cpu().numpy())

            if "REGRESS" in self.task_type:
                if "rank_ordinal" in self.loss_type:
                    logits, probas = out_net[0], out_net[1]
                    predict_levels = probas > 0.5
                    pred = torch.sum(predict_levels, dim=1)
                    return dict(logit_r=pred.cpu().numpy(),
                                true=true.cpu().numpy())
                if "rank_dorn" in self.loss_type:
                    pred, softmax = net(imgs)
                    return dict(logit_r=pred.cpu().numpy(),
                                true=true.cpu().numpy())
                if "soft_label" in self.loss_type:
                    logit_regress = (self.nr_classes - 1) * out_net
                    return dict(logit_r=logit_regress.cpu().numpy(),
                                true=true.cpu().numpy())
                if "FocalOrdinal" in self.loss_type:
                    logit_regress = out_net
                    pred = count_pred(logit_regress)
                    return dict(logit_r=pred.cpu().numpy(),
                                true=true.cpu().numpy())
                else:
                    logit_regress = out_net
                    return dict(logit_r=logit_regress.cpu().numpy(),
                                true=true.cpu().numpy())

            if "MULTI" in self.task_type:
                logit_class, logit_regress = out_net[0], out_net[1]
                prob = nn.functional.softmax(logit_class, dim=-1)
                return dict(logit_c=prob.cpu().numpy(),
                            logit_r=logit_regress.cpu().numpy(),
                            true=true.cpu().numpy())

    ####
    def run_once(self, fold_idx):

        log_dir = self.log_dir
        check_manual_seed(self.seed)
        train_augmentors = self.train_augmentors()
        infer_augmentors = self.infer_augmentors()  # HACK at has_aux

        if self.exp_args.dataset == "kbsmc_colon":
            data_func = "prepare_colon_tma_1024_data"
        elif self.exp_args.dataset == "uhu_prostate":
            data_func = "prepare_prostate_uhu_data"
        elif self.exp_args.dataset == "gastric":
            data_func = "prepare_gastric_data"
        elif self.exp_args.dataset == "bladder":
            data_func = "prepare_bladder_data"

        train_pairs, valid_pairs, test_pairs = getattr(dataset, (data_func))(data_root_dir=args.image_path)
        train_dataset = dataset.DatasetSerialImgsAndGraph(train_pairs[:], has_aux=False,
                                    shape_augs=iaa.Sequential(train_augmentors[0]),
                                    input_augs=iaa.Sequential(train_augmentors[1]),
                                    data_root_dir=self.exp_args.image_path,
                                    graph_root_dir=self.exp_args.spatially_constrained_graph_path,
                                    dataset_name=self.exp_args.dataset)

        infer_dataset = dataset.DatasetSerialImgsAndGraph(valid_pairs[:], has_aux=False,
                                    shape_augs=iaa.Sequential(infer_augmentors[0]),
                                    data_root_dir=self.exp_args.image_path,
                                    graph_root_dir=self.exp_args.spatially_constrained_graph_path,
                                    dataset_name=self.exp_args.dataset)
        
        test_dataset = dataset.DatasetSerialImgsAndGraph(test_pairs[:], has_aux=False,
                                    shape_augs=iaa.Sequential(infer_augmentors[0]),
                                    data_root_dir=self.exp_args.image_path,
                                    graph_root_dir=self.exp_args.spatially_constrained_graph_path,
                                    dataset_name=self.exp_args.dataset)

        train_loader = data.DataLoader(train_dataset,
                                       num_workers=self.nr_procs_train,
                                       batch_size=self.train_batch_size,
                                       shuffle=True, drop_last=True)
        
        valid_loader = data.DataLoader(infer_dataset,
                                       num_workers=self.nr_procs_valid,
                                       batch_size=self.infer_batch_size,
                                       shuffle=False, drop_last=False)
        
        test_loader = data.DataLoader(test_dataset,
                                      num_workers=self.nr_procs_valid,
                                      batch_size=self.infer_batch_size,
                                      shuffle=False, drop_last=False)

        if self.logging:
            check_log_dir(log_dir)

        device = 'cuda'

        # Define your network here
        net = SCUBaNet(num_nodes=16, node_dim=512, embed_dim=768) # SCUBa-Net
        net = torch.nn.DataParallel(net).to(device)
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Num params:', pytorch_total_params)
        # optimizers
        optimizer = torch.optim.SGD(net.parameters(),
                                lr=3e-2,
                                momentum=0.9,
                                weight_decay=0)
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=10000)
        
        iters = self.nr_epochs * self.epoch_length
        trainer = Engine(lambda engine, batch: self.train_step(engine, net, batch, iters, scheduler, optimizer, device))
        valider = Engine(lambda engine, batch: self.infer_step(net, batch, device))
        test = Engine(lambda engine, batch: self.infer_step(net, batch, device))

        # assign output
        if "CLASS" in self.task_type:
            infer_output = ['logit_c', 'true']
        if "REGRESS" in self.task_type:
            infer_output = ['logit_r', 'true']
        if "MULTI" in self.task_type:
            infer_output = ['logit_c', 'logit_r', 'pred_c', 'pred_r', 'true']

        ##
        events = Events.EPOCH_COMPLETED
        if self.logging:
            @trainer.on(events)
            def save_chkpoints(engine):
                torch.save(net.state_dict(), self.log_dir + '/_net_' + str(engine.state.iteration) + '.pth')

        timer = Timer(average=True)
        timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
        timer.attach(valider, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
        timer.attach(test, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

        # attach running average metrics computation
        # decay of EMA to 0.95 to match tensorpack default
        # TODO: refactor this
        RunningAverage(alpha=0.95, output_transform=lambda x: x['acc']).attach(trainer, 'acc')
        RunningAverage(alpha=0.95, output_transform=lambda x: x['loss']).attach(trainer, 'loss')

        # attach progress bar
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=['loss'])
        pbar.attach(valider)
        pbar.attach(test)

        # writer for tensorboard logging
        tfwriter = None  # HACK temporary
        if self.logging:
            tfwriter = SummaryWriter(logdir=log_dir)
            json_log_file = log_dir + '/stats.json'
            with open(json_log_file, 'w') as json_file:
                json.dump({}, json_file)  # create empty file

        ### TODO refactor again
        log_info_dict = {
            'logging': self.logging,
            'optimizer': optimizer,
            'tfwriter': tfwriter,
            'json_file': json_log_file if self.logging else None,
            'nr_classes': self.nr_classes,
            'metric_names': infer_output,
            'infer_batch_size': self.infer_batch_size  # too cumbersome
        }
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  lambda engine: scheduler.step(engine.state.epoch - 1))  # to change the lr
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_train_ema_results, log_info_dict)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, inference, valider, 'valid', valid_loader, log_info_dict)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, inference, test, 'test', test_loader, log_info_dict)
        valider.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)
        test.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)

        # Setup is done. Now let's run the training
        # trainer.run(train_loader, self.nr_epochs)
        trainer.run(train_loader, self.nr_epochs, self.epoch_length)
        return

    ####
    def run(self):
        if self.cross_valid:
            for fold_idx in range(0, trainer.nr_fold):
                trainer.run_once(fold_idx)
        else:
            self.run_once(self.fold_idx)
        return

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run_info', type=str, default='REGRESS_rank_dorn',
                        help='CLASS, REGRESS, MULTI + loss, '
                             'loss ex: MULTI_mtmr, REGRESS_rank_ordinal, REGRESS_rank_dorn'
                             'REGRESS_FocalOrdinalLoss, REGRESS_soft_ordinal')
    parser.add_argument('--dataset', type=str, default='colon_tma', help='colon_tma, prostate_uhu')
    parser.add_argument('--seed', type=int, default=5, help='number')
    parser.add_argument('--alpha', type=int, default=5, help='number')

    parser.add_argument('--dataset', type=str, default="", help='kbsmc_colon, uhu_prostate, gastric, bladder')
    parser.add_argument('--image_path', type=str, default="", help='image path')
    parser.add_argument('--spatially_constrained_graph_path', type=str, default="", help='spatially constrained graph_path')

    args = parser.parse_args()

    trainer = Trainer(_args=args)
    if args.view:
        trainer.view_dataset()
        exit()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    trainer.run()