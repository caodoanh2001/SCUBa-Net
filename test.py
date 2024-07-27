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
from tqdm import tqdm
import math
import dataset as dataset
from config import Config
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, cohen_kappa_score
from models import *
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
    
class Tester(Config):
    def __init__(self, _args=None):
        super(Tester, self).__init__(_args=_args)
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

    def infer_step(self, net, batch, device):
        net.eval()  # infer mode
        imgs_cpu, graphs_cpu, adjs_cpu, true_cpu = batch

        # push data to GPUs and convert to float32
        imgs = imgs_cpu.to(device).float()
        graphs = graphs_cpu.to(device).float()
        true = true_cpu.to(device).long()  # not one-hot
        adjs = adjs_cpu.to(device).float()

        # -----------------------------------------------------------
        with torch.no_grad():  # dont compute gradient
            # out_net = net(imgs, graphs, adjs)
            out_net = net(imgs, graphs, adjs)
            logit_class = out_net
            prob = nn.functional.softmax(logit_class, dim=-1)
            return dict(logit_c=prob.cpu().numpy(),  # from now prob of class task is called by logit_c
                        true=true.cpu().numpy())

    ####
    def run_once(self, fold_idx):

        log_dir = self.log_dir
        check_manual_seed(self.seed)
        # --------------------------- Dataloader
        infer_augmentors = self.infer_augmentors()  # HACK at has_aux
        if self.exp_args.dataset == "kbsmc_colon":
            data_func = "prepare_colon_tma_1024_data"
        elif self.exp_args.dataset == "kbsmc_colon_test_2":
            data_func = "prepare_colon_tma_data_test_2"
        elif self.exp_args.dataset == "uhu_prostate":
            data_func = "prepare_prostate_uhu_data"
        elif self.exp_args.dataset == "gastric":
            data_func = "prepare_gastric_data"
        elif self.exp_args.dataset == "bladder":
            data_func = "prepare_bladder_data"

        _, _, test_pairs = getattr(dataset, (data_func))(data_root_dir=args.image_path)
       
        test_dataset = dataset.DatasetSerialImgsAndGraph(test_pairs[:], has_aux=False,
                                    shape_augs=iaa.Sequential(infer_augmentors[0]),
                                    data_root_dir=args.image_path,
                                    graph_root_dir=args.spatially_constrained_graph_path,
                                    dataset_name=args.dataset)
        
        test_loader = data.DataLoader(test_dataset,
                                      num_workers=self.nr_procs_valid,
                                      batch_size=self.infer_batch_size,
                                      shuffle=False, drop_last=False)

        device = 'cuda'

        # Define network and load checkpoint path
        PATH_model = self.exp_args.path_model # checkpoint path
        net = SCUBaNet(num_nodes=16, node_dim=512, embed_dim=768)
        net = torch.nn.DataParallel(net).to(device)
        checkpoint = torch.load(PATH_model)
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Num params:', pytorch_total_params)
        net.load_state_dict(checkpoint, strict=True)
        net.eval()

        logits_c = []
        trues = []

        # Evaluating
        with tqdm(desc='Evaluation', unit='it', total=len(test_loader)) as pbar:
            for it, (imgs, graphs, adjs, gts) in enumerate(iter(test_loader)):
                results = self.infer_step(net, (imgs, graphs, adjs, gts), device)
                logits_c.append(results['logit_c'])
                trues.append(results['true'])
                pbar.update()

        logits_c = np.concatenate(logits_c, axis=0)
        trues = np.concatenate(trues)
        preds_c = np.argmax(logits_c, axis=-1)

        if max(trues) == 4: trues -= 1 # For KBSMC test 2

        print('Precision: ', precision_score(trues, preds_c, average='macro'))
        print('Recall: ', recall_score(trues, preds_c, average='macro'))
        print('F1: ', f1_score(trues, preds_c, average='macro'))
        print('Accuracy: ', accuracy_score(trues, preds_c))
        print('Kw:', cohen_kappa_score(trues, preds_c, weights='quadratic'))
        print('Confusion matrix: ')
        print(confusion_matrix(trues, preds_c))
        return

    ####
    def run(self):
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

    # Additional args for GCN experiments
    parser.add_argument('--dataset', type=str, default="", help='kbsmc_colon, kbsmc_colon_test_2, uhu_prostate, gastric, bladder')
    parser.add_argument('--image_path', type=str, default="", help='image path')
    parser.add_argument('--spatially_constrained_graph_path', type=str, default="", help='spatially constrained graph_path')
    parser.add_argument('--path_model', type=str, default="", help='str')
    args = parser.parse_args()

    tester = Tester(_args=args)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    tester.run()