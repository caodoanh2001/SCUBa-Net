from data_simclr import DataSetWrapper, DataSetWrapperProstate, DataSetWrapperGastric
import torch.nn.functional as F
import torch.nn as nn
import yaml
import torch
import os
import numpy as np
from tqdm import tqdm
import timm

class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
    
class EffSimCLR(nn.Module):
    def __init__(self, out_dim):
        super(EffSimCLR, self).__init__()
        net = timm.create_model('efficientnet_b0', pretrained="efficientnet_b0_ra-3dd342df.pth")
        num_ftrs = net.classifier.in_features
        self.features = nn.Sequential(*list(net.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x

config = yaml.load(open("config_clr.yaml", "r"), Loader=yaml.FullLoader)

# colon
if config['dataset_name'] == 'colon':
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

# prostate
elif config['dataset_name'] == 'prostate':
    dataset = DataSetWrapperProstate(config['batch_size'], **config['dataset'])

# gastric
elif config['dataset_name'] == 'gastric':
    dataset = DataSetWrapperGastric(config['batch_size'], **config['dataset'])

# bladder
elif config['dataset_name'] == 'bladder':
    dataset = DataSetWrapperGastric(config['batch_size'], **config['dataset'])


train_loader, valid_loader = dataset.get_data_loaders()
device = 'cuda:0'
log_dir = config['log_dir']
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

model = EffSimCLR(out_dim=config['out_dim'])
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=eval(config['weight_decay']))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0,
                                                               last_epoch=-1)
model_checkpoints_folder = os.path.join(log_dir, 'checkpoints')
if not os.path.exists(model_checkpoints_folder):
    os.mkdir(model_checkpoints_folder)

criterion = NTXentLoss(device, config['batch_size'], **config['loss'])
def _step(model, xis, xjs, n_iter, criterion):
    # get the representations and the projections
    _, zis = model(xis)  # [N,C]

    # get the representations and the projections
    _, zjs = model(xjs)  # [N,C]

    # normalize projection feature vectors
    zis = F.normalize(zis, dim=1)
    zjs = F.normalize(zjs, dim=1)

    loss = criterion(zis, zjs)
    return loss 

def _validate(model, valid_loader):
    # validation steps
    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        counter = 0
        with tqdm(desc='Epoch %d - evaluation', unit='it', total=len(valid_loader)) as pbar:
            for (xis, xjs) in valid_loader:
                xis = xis.to(device)
                xjs = xjs.to(device)
                loss = _step(model, xis, xjs, counter, criterion)
                valid_loss += loss.item()
                counter += 1
                pbar.set_postfix(loss=loss.item())
                pbar.update()
            valid_loss /= counter
    model.train()
    return valid_loss

n_iter = 0
valid_n_iter = 0
best_valid_loss = np.inf
for epoch_counter in range(config['epochs']):
    with tqdm(desc='Epoch: ' + str(epoch_counter) + ' - training', unit='it', total=len(train_loader)) as pbar:
        for (xis, xjs) in train_loader:
            optimizer.zero_grad()
            xis = xis.to(device)
            xjs = xjs.to(device)
            loss = _step(model, xis, xjs, n_iter, criterion)
            # if n_iter % config['log_every_n_steps'] == 0:
            #     print("[%d/%d] step: %d train_loss: %.3f" % (epoch_counter, config['epochs'], n_iter, loss))
            loss.backward()
            optimizer.step()
            n_iter += 1
            pbar.set_postfix(loss=loss.item())
            pbar.update()

        # validate the model if requested
        if epoch_counter % config['eval_every_n_epochs'] == 0:
            valid_loss = _validate(model, valid_loader)
            # print("[%d/%d] val_loss: %.3f" % (epoch_counter, config['epochs'], valid_loss))
            if valid_loss < best_valid_loss:
                # save the model weights
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                # print('saved')
            valid_n_iter += 1

        # warmup for the first 10 epochs
        if epoch_counter >= 10:
            scheduler.step()