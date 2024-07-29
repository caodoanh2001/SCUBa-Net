import torch.nn.functional as F
import torch.nn as nn
import sys
import timm
import glob
import os
import torch
import os
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from data_simclr import ToPIL
from PIL import Image
import argparse
import yaml

device = 'cuda:0'
PATCH_SIZE = 256
STEP_SIZE = 256
MAX_IMG_SIZE = 1024

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to folder of pathology images', default="")
parser.add_argument('--data_graph_path', type=str, help='Path to folder including saved sub-patch images', default="")

config = yaml.load(open("config_clr.yaml", "r"), Loader=yaml.FullLoader)
log_dir = config['log_dir']
model_checkpoints_folder = os.path.join(log_dir, 'checkpoints')
ckpt_path = os.path.join(model_checkpoints_folder, 'model.pth')

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

def build_graph(args):
    model = EffSimCLR(out_dim=512)
    model = model.to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    data_transforms = transforms.Compose([ToPIL(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),])
    data_path = args.data_path
    data_output_path = args.data_output_path
    patch_size, step_size = PATCH_SIZE, STEP_SIZE

    for path in [data_output_path]:
        os.makedirs(path, exist_ok=True)

    def crop_image_into_patches(image, patch_size, step_size):
        _, height, width = image.shape
        patches = []
        
        for i in range(0, height - patch_size + 1, step_size):
            for j in range(0, width - patch_size + 1, step_size):
                patch = image[:, i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        
        return patches

    for folder in tqdm(glob.glob(data_path + "/*")):
        out_sub_dir = os.path.join(data_output_path, folder.split('/')[-1]) + '/'
        if not os.path.exists(out_sub_dir):
            os.mkdir(out_sub_dir)
        for img_path in glob.glob(os.path.join(folder + '/*')):
            img = Image.open(img_path)
            img = img.resize((MAX_IMG_SIZE, MAX_IMG_SIZE))
            img = transforms.functional.to_tensor(img)
            sample = data_transforms(img).to(device)
            patches = crop_image_into_patches(sample, patch_size, step_size)
            stacked_feats = []
            for patch in patches:
                with torch.no_grad():
                    _, feat = model(patch.unsqueeze(0))
                stacked_feats.append(feat.unsqueeze(0))
            stacked_feats = torch.cat(stacked_feats, dim=0).cpu().detach().numpy()
            np.save(os.path.join(out_sub_dir, os.path.basename(img_path).split('.')[0] + '.npy'), stacked_feats)

if __name__ == '__main__': 
    args = parser.parse_args()
    os.makedirs(args.data_graph_path, exist_ok=True)
    build_graph(args.data_path, args.data_graph_path)