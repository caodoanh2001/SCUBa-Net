import os
import glob
import cv2
from tqdm import tqdm
import argparse

PATCH_SIZE = 256
STEP_SIZE = 256
MAX_IMG_SIZE = 1024

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to folder of pathology images', default="")
parser.add_argument('--data_output_path', type=str, help='Path to folder including saved sub-patch images', default="")

"""
data_path should follow below structure:

DATASET
|_CLASS1
|__*.jpg/png
|_CLASS2
|__*.jpg/png
...
|_CLASSN
|__*.jpg/png
"""

def crop_image_into_patches(image, patch_size, step_size):
    height, width, _ = image.shape
    patches = []
    
    for i in range(0, height - patch_size + 1, step_size):
        for j in range(0, width - patch_size + 1, step_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)

    return patches

def patching(data_path, data_output_path):
    patch_size = PATCH_SIZE
    step_size = STEP_SIZE
    for img_path in tqdm(glob.glob(os.path.join(data_path + '/**/**'))):
        image = cv2.imread(img_path)
        image = cv2.resize(image, (MAX_IMG_SIZE, MAX_IMG_SIZE))
        patches = crop_image_into_patches(image, patch_size, step_size)
        save_dir = data_output_path + '/' + img_path.split('/')[-2]
        os.makedirs(save_dir, exist_ok=True)
        for i, patch in enumerate(patches):
            cv2.imwrite(save_dir + "/" + os.path.basename(img_path).split('.')[0] + '_' + str(i) + '.jpg', patch)

if __name__ == '__main__': 
    args = parser.parse_args()
    os.makedirs(args.data_output_path, exist_ok=True)
    patching(args.data_path, args.data_output_path)