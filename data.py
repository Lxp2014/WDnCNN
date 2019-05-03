import numpy as np
import torch as t
import torch.utils.data as data
from utils import *
from random import sample
import cv2
import os
import os.path
from cfg import par
import random


class train_dataset_color(data.Dataset):
    def __init__(self, root='/home/rick/database/Denoising/Waterloo/', pic=4744 + 1000):
        print('Loading Training Data...')
        self.root = root
        self.pic = sample(list(range(pic)), 1000)
        target = []
        for i in self.pic:
            if i < 4744:
                self.root = '/home/rick/database/Denoising/Waterloo/'
                if i < 9:
                    idx = '0000' + str(i + 1)
                elif i < 99:
                    idx = '000' + str(i+1)
                elif i < 999:
                    idx = '00' + str(i+1)
                else:
                    idx = '0' + str(i + 1)
                img_name = idx + '.bmp'
            else:
                self.root = '/home/rick/database/Denoising/ImageNet_val_part/'
                idx = i - 4744 + 1
                if idx < 10:
                    idx = '0000000' + str(idx)
                elif idx < 100:
                    idx = '000000' + str(idx)
                elif idx < 1000:
                    idx = '00000' + str(idx)
                else:
                    idx = '0000' + str(idx)
                img_name = 'ILSVRC2012_val_' + idx + '.JPEG'
            img_dir = os.path.join(self.root, img_name)
            img = cv2.imread(img_dir, cv2.IMREAD_COLOR).astype(np.float32)  # load color images (W H C)
            patch_t = patch_generator_color(img, par)  # C W H
            target.extend(patch_t)
        self.target = sample(target, 128*2000)
        print('~~~~~~~~Data Loading Succeeds ^_^ ~~~~~~~~')

    def __getitem__(self, item):
        target = self.target[item]
        seed = random.randint(0, 2 ** 32 - 1)
        if seed == 0:
            print(seed)
        np.random.seed(seed=seed)
        sigma = np.random.uniform(par.sigma[0], par.sigma[1])
        noise = np.random.normal(0, sigma / 255.0, target.shape)
        noisy = target + noise
        noisy = noisy.astype(np.float32)

        tensor_n, tensor_t = color_tensor_generator(noisy, target)
        nlm = np.tile(sigma / 255.0, tensor_n[0].shape).astype(np.float32)  # noise level map
        tensor_n.append(nlm)
        noisy = np.stack(tensor_n, axis=0)
        target = np.stack(tensor_t, axis=0)

        target = t.from_numpy(target.copy())
        noisy = t.from_numpy(noisy.copy())
        return noisy, target

    def __len__(self):
        return len(self.target)


class train_dataset_gray(data.Dataset):
    def __init__(self, root='/home/rick/database/Denoising/Waterloo/', pic=4744 + 400 + 600):
        print('Loading Training Data...')
        self.root = root
        self.pic = sample(list(range(pic)), 1000)
        target = []
        for i in self.pic:
            if i < 4744:
                self.root = '/home/rick/database/Denoising/Waterloo/'
                if i < 9:
                    idx = '0000' + str(i + 1)
                elif i < 99:
                    idx = '000' + str(i+1)
                elif i < 999:
                    idx = '00' + str(i+1)
                else:
                    idx = '0' + str(i + 1)
                img_name = idx + '.bmp'
            elif i < 4744+400:
                self.root = '/home/rick/database/Denoising/trainset400/'
                idx = i - 4744 + 1
                if idx < 10:
                    idx = '00' + str(idx)
                elif idx < 100:
                    idx = '0' + str(idx)
                else:
                    idx = str(idx)
                img_name = 'test_' + idx + '.png'
            else:
                self.root = '/home/rick/database/Denoising/ImageNet_val_part/'
                idx = i - 4744 - 400 + 1
                if idx < 10:
                    idx = '0000000' + str(idx)
                elif idx < 100:
                    idx = '000000' + str(idx)
                else:
                    idx = '00000' + str(idx)
                img_name = 'ILSVRC2012_val_' + idx + '.JPEG'
            img_dir = os.path.join(self.root, img_name)
            img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE).astype(np.float32)# load gray images (W H)
            patch_t = patch_generator_gray(img, par)  # W H
            target.extend(patch_t)
        self.target = sample(target, 128*2000)
        print('~~~~~~~~~~~~~~~~~~~~Data Loading Succeeds ^_^ ~~~~~~~~~~~~~~~~~~~~')

    def __getitem__(self, item):
        target = self.target[item]
        seed = random.randint(0,  2**32 - 1)
        np.random.seed(seed=seed)
        sigma = np.random.uniform(par.sigma[0], par.sigma[1])
        noise = np.random.normal(0, sigma / 255.0, target.shape)
        noisy = target + noise
        noisy = noisy.astype(np.float32).squeeze()

        tensor_n, tensor_t = gray_tensor_generator(noisy, target)
        nlm = np.tile(sigma / 255.0, tensor_n[0].shape).astype(np.float32)  # noise level map
        tensor_n.append(nlm)
        noisy = np.stack(tensor_n, axis=0)
        target = np.stack(tensor_t, axis=0)

        target = t.from_numpy(target.copy())
        noisy = t.from_numpy(noisy.copy())
        return noisy, target

    def __len__(self):
        return len(self.target)


class valid_dataset_color(data.Dataset):
    def __init__(self, root='/home/rick/database/Denoising/testset/', dataset_name=None):
        self.root = os.path.join(root, dataset_name)
        self.img_batch = []
        for root, dirs, files in os.walk(self.root):
            for i in range(len(files)):
                img = cv2.imread(self.root+'/'+files[i], cv2.IMREAD_COLOR).astype(np.float32) # W*H*C
                img = np.divide(img, 255).transpose(2, 0, 1) # C*W*H
                self.img_batch.append(img)

    def __getitem__(self, item):
        img = self.img_batch[item]
        np.random.seed(seed=0)  # for reproducibility
        noise = np.random.normal(0, par.test_sigma / 255.0, img.shape)
        img_n = img + noise
        img_n = img_n.astype(np.float32)

        tensor_n, _ = color_tensor_generator(img_n, img)
        nlm = np.tile(par.test_sigma / 255.0, tensor_n[0].shape).astype(np.float32)  # noise level map
        tensor_n.append(nlm)
        img_n = np.stack(tensor_n, axis=0)

        img = t.from_numpy(img.copy())
        img_n = t.from_numpy(img_n.copy())
        return img_n, img

    def __len__(self):
        return len(self.img_batch)


class valid_dataset_gray(data.Dataset):
    def __init__(self, root='/home/rick/database/Denoising/testset/', dataset_name=None):
        self.root = os.path.join(root, dataset_name)
        self.img_batch = []
        for root, dirs, files in os.walk(self.root):
            if dataset_name == 'Set12':
                files = ['01.png', '02.png', '03.png', '04.png', '05.png', '06.png', '07.png', '08.png',
                                '09.png', '10.png', '11.png', '12.png']
            for i in range(len(files)):
                img = cv2.imread(self.root+'/'+files[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
                img = np.divide(img, 255.0)
                self.img_batch.append(img)

    def __getitem__(self, item):
        img = self.img_batch[item]
        np.random.seed(seed=0)  # for reproducibility
        noise = np.random.normal(0, par.test_sigma / 255.0, img.shape)
        img_n = img + noise
        img_n = img_n.astype(np.float32).squeeze()

        tensor_n, _ = gray_tensor_generator(img_n, img)
        nlm = np.tile(par.test_sigma / 255.0, tensor_n[0].shape).astype(np.float32)  # noise level map
        tensor_n.append(nlm)
        img_n = np.stack(tensor_n, axis=0)

        img = t.from_numpy(img.copy())
        img_n = t.from_numpy(img_n.copy())
        return img_n, img

    def __len__(self):
        return len(self.img_batch)
    
