import os
import csv
import glob
import random
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from imgaug import augmenters as iaa
import torch

class DatasetSerialImgsAndGraph(data.Dataset):

    def __init__(self, pair_list, 
                 shape_augs=None, 
                 input_augs=None, 
                 has_aux=False, 
                 test_aux=False,
                 data_root_dir=None,
                 graph_root_dir=None,
                 dataset_name=None):
        self.test_aux = test_aux
        self.pair_list = pair_list
        self.shape_augs = shape_augs
        self.input_augs = input_augs
        self.dataset_name = dataset_name
        self.edges = {
            0: [1, 4, 5],
            1: [0, 2, 4, 5, 6],
            2: [1, 3, 5, 6, 7],
            3: [2, 6, 7],
            4: [0, 1, 5, 8, 9],
            5: [0, 1, 2, 4, 6, 8, 9, 10],
            6: [1, 2, 3, 5, 7, 9, 10, 11],
            7: [2, 3, 6, 10, 11],
            8: [4, 5, 9, 12, 13],
            9: [4, 5, 6, 8, 10, 12, 13, 14],
            10: [5, 6, 7, 9, 11, 13, 14, 15],
            11: [6, 7, 10, 14, 15],
            12: [8, 9, 13],
            13: [8, 9, 10, 12, 14],
            14: [9, 10, 11, 13, 15],
            15: [10, 11, 14],
        }
        self.data_root_dir = data_root_dir
        self.graph_root_dir = graph_root_dir

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        filename = pair[0]
        input_img = cv2.imread(filename)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_label = pair[1]
        # print(input_img.shape)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1., 1., 1.])
        ])

        if not self.test_aux:

            # shape must be deterministic so it can be reused
            if self.shape_augs is not None:
                shape_augs = self.shape_augs.to_deterministic()
                input_img = shape_augs.augment_image(input_img)

            # additional augmenattion just for the input
            if self.input_augs is not None:
                input_img = self.input_augs.augment_image(input_img)

            input_img = np.array(input_img).copy()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1., 1., 1.])
            ])

            out_img = np.array(transform(input_img)).transpose(1, 2, 0)
        else:
            out_img = []
            for idx in range(5):
                input_img_ = input_img.copy()
                if self.shape_augs is not None:
                    shape_augs = self.shape_augs.to_deterministic()
                    input_img_ = shape_augs.augment_image(input_img_)
                input_img_ = iaa.Sequential(self.input_augs[idx]).augment_image(input_img_)
                input_img_ = np.array(input_img_).copy()
                input_img_ = np.array(transform(input_img_)).transpose(1, 2, 0)
                out_img.append(input_img_)
        
        # Because of different directory structures for each dataset, we do below for convinience
        if self.dataset_name in ["kbsmc_colon", "uhu_prostate", "ubc_prostate"]:
            input_graph = np.load(filename.replace(self.data_root_dir, self.graph_root_dir).split('.')[0] + '.npy', allow_pickle=True)
        elif self.dataset_name == "kbsmc_colon_test_2":
            input_graph = np.load(self.graph_root_dir + '/' + filename + '.npy', allow_pickle=True) # CTestII
        elif self.dataset_name == "gastric":
            input_graph = np.load(self.graph_root_dir + '/' + os.path.basename(filename).split('.')[0] + '.npy', allow_pickle=True) # Gastric
        elif self.dataset_name == "bladder":
            # Bladder
            graph_path = filename.replace(self.data_root_dir, self.graph_root_dir).split('.')[0] + '.npy'
            graph_path = graph_path.split('/')
            graph_path = '/'.join(graph_path[:-3]) + '/' + graph_path[-1]
            input_graph = np.load(graph_path, allow_pickle=True)
        
        adj_s = np.zeros((input_graph.shape[0] + 1, input_graph.shape[0] + 1))
        for i in self.edges:
            for j in self.edges[i]:
                adj_s[i + 1][j + 1] = 1

        # only patch nodes have connections to [CLS] node
        for i in range(input_graph.shape[0]):
            adj_s[i + 1][0] = 1

        for i in range(input_graph.shape[0]):
            adj_s[0][i + 1] = 1

        return np.array(out_img), input_graph, adj_s, img_label

    def __len__(self):
        return len(self.pair_list)

def print_data_count(label_list):
    count = []
    for i in range(5):
        count.append(label_list.count(i))
    count.append(len(label_list))
    return count

def prepare_colon_tma_1024_data(data_root_dir=None):
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        cancer_test = False
        if cancer_test:
            file_list_bn = glob.glob(pathname.replace('*.jpg', '*0.jpg'))
            file_list = [elem for elem in file_list if elem not in file_list_bn]
            label_list = [int(file_path.split('_')[-1].split('.')[0])-1 for file_path in file_list]
        else:
            if parse_label:
                label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            else:
                label_list = [label_value for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    set_1010711 = load_data_info('%s/1010711/*.jpg' % data_root_dir)
    set_1010712 = load_data_info('%s/1010712/*.jpg' % data_root_dir)
    set_1010713 = load_data_info('%s/1010713/*.jpg' % data_root_dir)
    set_1010714 = load_data_info('%s/1010714/*.jpg' % data_root_dir)
    set_1010715 = load_data_info('%s/1010715/*.jpg' % data_root_dir)
    set_1010716 = load_data_info('%s/1010716/*.jpg' % data_root_dir)
    wsi_00016 = load_data_info('%s/wsi_00016/*.jpg' % data_root_dir, parse_label=True,
                               label_value=0)  # benign exclusively
    wsi_00017 = load_data_info('%s/wsi_00017/*.jpg' % data_root_dir, parse_label=True,
                               label_value=0)  # benign exclusively
    wsi_00018 = load_data_info('%s/wsi_00018/*.jpg' % data_root_dir, parse_label=True,
                               label_value=0)  # benign exclusively

    train_set = set_1010711 + set_1010712 + set_1010713 + set_1010715 + wsi_00016
    valid_set = set_1010716 + wsi_00018
    test_set = set_1010714 + wsi_00017

    # print dataset detail
    train_label = [train_set[i][1] for i in range(len(train_set))]
    val_label = [valid_set[i][1] for i in range(len(valid_set))]
    test_label = [test_set[i][1] for i in range(len(test_set))]
    
    print(print_data_count(train_label))
    print(print_data_count(val_label))
    print(print_data_count(test_label))
    
    return train_set, valid_set, test_set

def prepare_colon_tma_data(
        data_root_dir=None):
    
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))
    
    set_tma01 = load_data_info('%s/tma_01/*.npy' % data_root_dir)
    set_tma02 = load_data_info('%s/tma_02/*.npy' % data_root_dir)
    set_tma03 = load_data_info('%s/tma_03/*.npy' % data_root_dir)
    set_tma04 = load_data_info('%s/tma_04/*.npy' % data_root_dir)
    set_tma05 = load_data_info('%s/tma_05/*.npy' % data_root_dir)
    set_tma06 = load_data_info('%s/tma_06/*.npy' % data_root_dir)
    set_wsi01 = load_data_info('%s/wsi_01/*.npy' % data_root_dir)  # benign exclusively
    set_wsi02 = load_data_info('%s/wsi_02/*.npy' % data_root_dir)  # benign exclusively
    set_wsi03 = load_data_info('%s/wsi_03/*.npy' % data_root_dir)  # benign exclusively

    train_set = set_tma01 + set_tma02 + set_tma03 + set_tma05 + set_wsi01
    valid_set = set_tma06 + set_wsi03
    test_set = set_tma04 + set_wsi02

    # print dataset detail
    train_label = [train_set[i][1] for i in range(len(train_set))]
    val_label = [valid_set[i][1] for i in range(len(valid_set))]
    test_label = [test_set[i][1] for i in range(len(test_set))]

    print(print_data_count(train_label))
    print(print_data_count(val_label))
    print(print_data_count(test_label))
    
    return train_set, valid_set, test_set

def prepare_colon_tma_1024_data_test_1(data_root_dir=None):
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        cancer_test = False
        if cancer_test:
            file_list_bn = glob.glob(pathname.replace('*.jpg', '*0.jpg'))
            file_list = [elem for elem in file_list if elem not in file_list_bn]
            label_list = [int(file_path.split('_')[-1].split('.')[0])-1 for file_path in file_list]
        else:
            if parse_label:
                label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            else:
                label_list = [label_value for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    set_tma04 = load_data_info('%s/1010714/*.npy' % data_root_dir)
    set_wsi02 = load_data_info('%s/wsi_00017/*.npy' % data_root_dir)  # benign exclusively

    train_set = None
    valid_set = None
    test_set = set_tma04 + set_wsi02
    # print dataset detail
    test_label = [test_set[i][1] for i in range(len(test_set))]
    print(print_data_count(test_label))
    
    return train_set, valid_set, test_set

def prepare_colon_tma_data_test_2(
        data_root_dir=None):
    
    def load_data_info(pathname):
        print(pathname)
        file_list = glob.glob(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    test_set_2 = []
    for folder in glob.glob('%s/*' % data_root_dir):
        set_info = load_data_info('%s/**/*.npy' % folder)
        test_set_2.append(set_info)

    train_set = None
    valid_set = None
    test_set = test_set_2[0]
    for i in range(1, len(test_set_2)): test_set += test_set_2[i]
    test_label = [test_set[i][1] for i in range(len(test_set))]
    print(print_data_count(test_label))
    
    return train_set, valid_set, test_set

def prepare_prostate_uhu_data(data_root_dir=None):
    def load_data_info(pathname, parse_label=True, label_value=0, cancer_test=False):
        file_list = glob.glob(pathname)

        if cancer_test:
            file_list_bn = glob.glob(pathname.replace('*.jpg', '*0.jpg'))
            file_list = [elem for elem in file_list if elem not in file_list_bn]
            label_list = [int(file_path.split('_')[-1].split('.')[0])-1 for file_path in file_list]
        else:
            if parse_label:
                label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            else:
                label_list = [label_value for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    data_root_dir_train = f'{data_root_dir}/patches_train_750_v0/'
    data_root_dir_validation = f'{data_root_dir}/patches_validation_750_v0/'
    data_root_dir_test = f'{data_root_dir}/patches_test_750_v0/'

    train_set_111 = load_data_info('%s/ZT111*/*.jpg' % data_root_dir_train)
    train_set_199 = load_data_info('%s/ZT199*/*.jpg' % data_root_dir_train)
    train_set_204 = load_data_info('%s/ZT204*/*.jpg' % data_root_dir_train)
    valid_set = load_data_info('%s/ZT76*/*.jpg' % data_root_dir_validation)
    test_set = load_data_info('%s/patho_1/*/*.jpg' % data_root_dir_test)

    train_set = train_set_111 + train_set_199 + train_set_204
    return train_set, valid_set, test_set

def prepare_prostate_ubc_data(data_root_dir=None):
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        cancer_test = False
        if cancer_test:
            file_list_bn = glob.glob(pathname.replace('*.jpg', '*0.jpg'))
            file_list = [elem for elem in file_list if elem not in file_list_bn]
            label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            label_dict = {2: 0, 3: 1, 4: 2}
            label_list = [label_dict[k] for k in label_list]
        else:
            if parse_label:
                label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            else:
                label_list = [label_value for file_path in file_list]
            label_dict = {0: 0, 2: 1, 3: 2, 4: 3}
            # import pdb; pdb.set_trace()
            label_list = [label_dict[k] for k in label_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    # assert fold_idx < 3, "Currently only support 5 fold, each fold is 1 TMA"

    data_root_dir_train_ubc = data_root_dir
    test_set_ubc = load_data_info('%s/*/*.jpg' % data_root_dir_train_ubc)
    return test_set_ubc

def print_number_of_sample(data_set, prefix):
    def fill_empty_label(label_dict):
        for i in range(max(label_dict.keys()) + 1):
            if label_dict[i] != 0:
                continue
            else:
                label_dict[i] = 0
        return dict(sorted(label_dict.items()))

    data_label = [data_set[i][1] for i in range(len(data_set))]
    d = Counter(data_label)
    d = fill_empty_label(d)
    print("%-7s" % prefix, d)
    data_label = [d[key] for key in d.keys()]

    return data_label

def load_gastric(csv_path, data_dir, data_dir_2, gt_list, nr_claases, down_sample=True):
    import pandas as pd
    def loader(path_list, data_root_dir, gt_list, nr_claases):
        file_list = []
        for tma_name in path_list:
            pathname = glob.glob(f'{data_root_dir}/{tma_name}/*.jpg')
            file_list.extend(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        label_list = [gt_list[i] for i in label_list]
        list_out = list(zip(file_list, label_list))

        list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] < nr_claases]
        return list_out

    df = pd.read_csv(csv_path).iloc[:, :3]
    train_list = list(df.query('Task == "train"')['WSI'])
    valid_list = list(df.query('Task == "val"')['WSI'])
    test_list = list(df.query('Task == "test"')['WSI'])
    train_set = loader(train_list, data_dir, gt_list, nr_claases)
    
    import tqdm
    import pickle
    if down_sample:
        train_normal = [train_set[i] for i in range(len(train_set)) if train_set[i][1] == 0]
        train_tumor = [train_set[i] for i in range(len(train_set)) if train_set[i][1] != 0]
        random.Random(42).shuffle(train_normal)
        train_normal = train_normal[: len(train_tumor) // 3]
        train_set = train_normal + train_tumor

    valid_set = loader(valid_list, data_dir_2, gt_list, nr_claases)
    test_set = loader(test_list, data_dir_2, gt_list, nr_claases)

    return train_set, valid_set, test_set

def prepare_gastric_data(data_root_dir=None, nr_classes=4, csv_her02=None, csv_addition=None, data_her_dir=None, 
                         data_her_dir_2=None, data_add_dir=None, data_add_dir_2=None):
    """ 8 classes in total only choose 5"""

    if nr_classes == 3:
        gt_train_local = {1: 4,  # "BN", #0
                          2: 4,  # "BN", #0
                          3: 0,  # "TW", #2
                          4: 1,  # "TM", #3
                          5: 2,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 4,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 4  # "LVI", #ignore
                          }
    elif nr_classes == 4:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 4,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 4  # "LVI", #ignore
                          }
    elif nr_classes == 5:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 8,  # "TLS", #1
                          7: 8,  # "papillary", #5
                          8: 8,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }
    elif nr_classes == 6:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 2,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 2,  # "TP", #4
                          6: 1,  # "TLS", #1
                          7: 3,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 5,  # "signet", #7
                          10: 5,  # "poorly", #7
                          11: 6  # "LVI", #ignore
                          }
    elif nr_classes == 8:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 2,  # "TW", #2
                          4: 3,  # "TM", #3
                          5: 4,  # "TP", #4
                          6: 1,  # "TLS", #1
                          7: 5,  # "papillary", #5
                          8: 6,  # "Mucinous", #6
                          9: 7,  # "signet", #7
                          10: 7,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }
    elif nr_classes == 10:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 5,  # "papillary", #5
                          8: 6,  # "Mucinous", #6
                          9: 7,  # "signet", #7
                          10: 8,  # "poorly", #7
                          11: 9  # "LVI", #ignore
                          }
    else:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 8,  # "TLS", #1
                          7: 8,  # "papillary", #5
                          8: 5,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }

    train_set, valid_set, test_set = load_gastric(csv_her02, data_her_dir, data_her_dir_2, gt_train_local, nr_classes)
    train_set_add, valid_set_add, test_set_add = load_gastric(csv_addition, data_add_dir, data_add_dir_2, gt_train_local, nr_classes, down_sample=False)
    train_set += train_set_add
    valid_set += valid_set_add
    test_set += test_set_add

    print_number_of_sample(train_set, 'Train')
    print_number_of_sample(valid_set, 'Valid')
    print_number_of_sample(test_set, 'Test')

    return train_set, valid_set, test_set

def prepare_bladder_data(data_root_dir=None, nr_classes=3):
    """ Bladder """
    
    train_set = [(item, 1) for item in glob.glob(data_root_dir + '/train/img/1/*')]
    train_set += [(item, 2) for item in glob.glob(data_root_dir + '/train/img/2/*')]
    train_set += [(item, 0) for item in glob.glob(data_root_dir + '/train/img/3/*')]
    
    valid_set = [(item, 1) for item in glob.glob(data_root_dir + '/val/img/1/*')]
    valid_set += [(item, 2) for item in glob.glob(data_root_dir + '/val/img/2/*')]
    valid_set += [(item, 0) for item in glob.glob(data_root_dir + '/val/img/3/*')]

    test_set = [(item, 1) for item in glob.glob(data_root_dir + '/test/img/1/*')]
    test_set += [(item, 2) for item in glob.glob(data_root_dir + '/test/img/2/*')]
    test_set += [(item, 0) for item in glob.glob(data_root_dir + '/test/img/3/*')]
    
    print_number_of_sample(train_set, 'Train')
    print_number_of_sample(valid_set, 'Valid')
    print_number_of_sample(test_set, 'Test')
    
    return train_set, valid_set, test_set