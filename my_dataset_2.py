# -*- coding: utf-8 -*
# @Time : 2022/10/3 20:38
# Author: Kunlin Yang
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

# Vertical flip
def flip_v(source, target):
    """Perform the flip_v operation.

    Args:
        source (Any): Description.
        target (Any): Description.

    Returns:
        Any: Result.
    """
    test_h = np.copy(source)
    test_h2 = np.copy(target)
    return test_h[::-1], test_h2[::-1]

# Rotate 180°
def flip180(arr, target):
    """Perform the flip180 operation.

    Args:
        arr (Any): Description.
        target (Any): Description.

    Returns:
        Any: Result.
    """
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)

    new_arr2 = target.reshape(target.size)
    new_arr2 = new_arr2[::-1]
    new_arr2 = new_arr2.reshape(target.shape)

    return new_arr, new_arr2

# Horizontal flip
def flip_h(arr, target):
    """Perform the flip_h operation.

    Args:
        arr (Any): Description.
        target (Any): Description.

    Returns:
        Any: Result.
    """
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    test_v = new_arr[::-1]

    new_arr2 = target.reshape(target.size)
    new_arr2 = new_arr2[::-1]
    new_arr2 = new_arr2.reshape(target.shape)
    test_v2 = new_arr2[::-1]

    return test_v, test_v2

# Rotate 270° (i.e., 90° counter‑clockwise)

def flip90_left(arr, target):
    """Perform the flip90_left operation.

    Args:
        arr (Any): Description.
        target (Any): Description.

    Returns:
        Any: Result.
    """
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]

    new_arr2 = np.transpose(target)
    new_arr2 = new_arr2[::-1]
    return new_arr, new_arr2

# Rotate 90°
def flip90_right(arr, target):
    """Perform the flip90_right operation.

    Args:
        arr (Any): Description.
        target (Any): Description.

    Returns:
        Any: Result.
    """
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
    """Class Numpy_Transform.

    Notes:
        Auto-generated documentation. Please refine as needed.
    """
    def __init__(self, mode='train', probability=0.5):
        """Initialize the instance.

        Args:
            mode (str): Description.
            probability (Any): Description.

        Returns:
            Any: Result.
        """
        self.mode = mode
        self.probability = probability

    def __call__(self, sample, target):
        """Perform the __call__ operation.

        Args:
            sample (Any): Description.
            target (Any): Description.

        Returns:
            Any: Result.
        """
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
    """Class DatasetFromFolder.

    Notes:
        Auto-generated documentation. Please refine as needed.
    """
    def __init__(self, image_dir, direction):
        """Initialize the instance.

        Args:
            image_dir (str): Description.
            direction (str): Description.

        Returns:
            Any: Result.
        """
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
        """Perform the __getitem__ operation.

        Args:
            index (Tensor): Description.

        Returns:
            Any: Result.
        """
        a = np.load(join(self.a_path, self.image_filenames[index]))
        b = np.load(join(self.b_path, self.image_filenames[index]))
        # a = a.resize((286, 286), Image.BICUBIC)
        # b = b.resize((286, 286), Image.BICUBIC)
        # Data augmentation，Mind the tensor shape/format
        a, b = self.transform(a, b)

        # a_max = np.max(a)
        # a_min = np.min(a)
        a = a  / 350

        # Normalize HR tensor
        # b_max = np.max(b)
        # b_min = np.min(b)
        b = b / 350

        # Convert to tensor
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
        """Perform the __len__ operation.

        Args:
            None

        Returns:
            Any: Result.
        """
        return len(self.image_filenames)
