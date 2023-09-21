# -*- coding: utf-8 -*
# @Time : 2022/10/3 20:38
# @Author : 杨坤林
# @File : my_dataset_2.py
# @Software : PyCharm


from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

# from utils import is_image_file, load_img


# 垂直翻转
def flip_v(source, target):
    test_h = np.copy(source)
    test_h2 = np.copy(target)
    return test_h[::-1], test_h2[::-1]

# 旋转180度
def flip180(arr, target):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)

    new_arr2 = target.reshape(target.size)
    new_arr2 = new_arr2[::-1]
    new_arr2 = new_arr2.reshape(target.shape)

    return new_arr, new_arr2

# 水平翻转
def flip_h(arr, target):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    test_v = new_arr[::-1]

    new_arr2 = target.reshape(target.size)
    new_arr2 = new_arr2[::-1]
    new_arr2 = new_arr2.reshape(target.shape)
    test_v2 = new_arr2[::-1]

    return test_v, test_v2

# 旋转270度（逆时针旋转90度）

def flip90_left(arr, target):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]

    new_arr2 = np.transpose(target)
    new_arr2 = new_arr2[::-1]
    return new_arr, new_arr2

# 旋转90度
def flip90_right(arr, target):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]

    new_arr2 = target.reshape(target.size)
    new_arr2 = new_arr2[::-1]
    new_arr2 = new_arr2.reshape(target.shape)
    new_arr2 = np.transpose(new_arr2)[::-1]

    return new_arr, new_arr2



class Numpy_Transform(object):
    def __init__(self, mode='train', probability=0.5):
        self.mode = mode
        self.probability = probability


    def __call__(self, sample, target):
        if self.mode == 'train':
            if round(np.random.uniform(0, 1), 1) <= self.probability:
                image1, image2 = sample, target
                image1, image2 = random.choice([flip_v,flip180, flip_h, flip90_right, flip90_left])(image1, image2)
                return image1, image2
            else:
                return sample, target

        if self.mode == 'test' or self.mode == 'infer':
            return sample, target




class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "b1")
        self.b_path = join(image_dir, "mw")
        self.image_filenames = [x for x in listdir(self.a_path)]

        self.transform = Numpy_Transform(probability=0.5)
        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        #
        # self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a = np.load(join(self.a_path, self.image_filenames[index]))
        b = np.load(join(self.b_path, self.image_filenames[index]))
        # a = a.resize((286, 286), Image.BICUBIC)
        # b = b.resize((286, 286), Image.BICUBIC)
        # 数据增强，注意格式
        a, b = self.transform(a, b)

        # a_max = np.max(a)
        # a_min = np.min(a)
        a = a  / 350

        # hr 归一化
        # b_max = np.max(b)
        # b_min = np.min(b)
        b = b / 350

        # 转化为tensor格式
        a = torch.from_numpy(np.expand_dims(a, 0))
        b = torch.from_numpy(np.expand_dims(b, 0))



        # a = transforms.ToTensor()(a)
        # b = transforms.ToTensor()(b)
        # w_offset = random.randint(0, max(0, 286 - 256 - 1))
        # h_offset = random.randint(0, max(0, 286 - 256 - 1))
        #
        # a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        # b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        #
        # a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        # b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)
        #
        # if random.random() < 0.5:
        #     idx = [i for i in range(a.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     a = a.index_select(2, idx)
        #     b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenames)
