import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets
import pandas as pd
from PIL import Image
from skimage import io, img_as_ubyte
import glob
np.random.seed(0)
import cv2
import numpy as np


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        # blur the image with a 50% chance
        prob = np.random.random_sample()
        if prob < 0.5:
#            print(self.kernel_size)
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample
    
class Dataset():
    def __init__(self, files_list, transform=None):
        self.files_list = files_list
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = Image.open(temp_path)
        img = transforms.functional.to_tensor(img)
        if self.transform:
            sample = self.transform(img)
        return sample

class ToPIL(object):
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img 

class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s, data_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)
        self.data_path = data_path

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()

        # train_set = ["tma_01", "tma_02", "tma_03", "tma_05", "wsi_01"]
        # valid_set = ["tma_06", "wsi_03"]

        train_set = ["1010711", "1010712", "1010713", "1010715", "wsi_00016"]
        valid_set = ["1010716", "wsi_00018"]
        train_list = []
        valid_list = []
        
        for s in train_set: train_list.extend(glob.glob(self.data_path + '/' + s + '/*'))
        for s in valid_set: valid_list.extend(glob.glob(self.data_path + '/' + s + '/*'))
        
        train_dataset = Dataset(files_list=train_list, transform=SimCLRDataTransform(data_augment))
        valid_dataset = Dataset(files_list=valid_list, transform=SimCLRDataTransform(data_augment))

        train_loader = self.get_train_validation_data_loaders(train_dataset)
        valid_loader = self.get_train_validation_data_loaders(valid_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([ToPIL(),
                                            #   transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
                                              transforms.Resize((self.input_shape[0],self.input_shape[1])),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.06 * self.input_shape[0])),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        train_sampler = SubsetRandomSampler(indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        return train_loader
    
class DataSetWrapperProstate(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s, data_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)
        self.data_path = data_path

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()

        train_set = ["ZT111*", "ZT199*", "ZT204*"]
        valid_set = ["ZT76*"]
        train_list = []
        valid_list = []

        for s in train_set: train_list.extend(glob.glob(self.data_path + '/patches_train_750_v0/' + s + '/*'))
        for s in valid_set: valid_list.extend(glob.glob(self.data_path + '/patches_validation_750_v0/' + s + '/*'))
        
        train_dataset = Dataset(files_list=train_list, transform=SimCLRDataTransform(data_augment))
        valid_dataset = Dataset(files_list=valid_list, transform=SimCLRDataTransform(data_augment))

        train_loader = self.get_train_validation_data_loaders(train_dataset)
        valid_loader = self.get_train_validation_data_loaders(valid_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([ToPIL(),
                                            #   transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
                                              transforms.Resize((self.input_shape[0],self.input_shape[1])),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.06 * self.input_shape[0])),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        train_sampler = SubsetRandomSampler(indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        return train_loader

class DataSetWrapperGastric(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s, data_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)
        self.data_path = data_path

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()

        train_list = glob.glob(self.data_path + '/train/*')
        valid_list = glob.glob(self.data_path + '/val/*')
        train_dataset = Dataset(files_list=train_list, transform=SimCLRDataTransform(data_augment))
        valid_dataset = Dataset(files_list=valid_list, transform=SimCLRDataTransform(data_augment))

        train_loader = self.get_train_validation_data_loaders(train_dataset)
        valid_loader = self.get_train_validation_data_loaders(valid_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([ToPIL(),
                                            #   transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
                                              transforms.Resize((self.input_shape[0],self.input_shape[1])),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.06 * self.input_shape[0])),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        train_sampler = SubsetRandomSampler(indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        return train_loader

class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj