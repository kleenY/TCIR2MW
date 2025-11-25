from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    """Perform the tensor2im operation.

    Args:
        image_tensor (Tensor): Description.
        imtype (Any): Description.

    Returns:
        Any: Result.
    """
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    """Perform the diagnose_network operation.

    Args:
        net (Any): Description.
        name (str): Description.

    Returns:
        Any: Result.
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def save_image(image_numpy, image_path):
    """Perform the save_image operation.

    Args:
        image_numpy (Any): Description.
        image_path (str): Description.

    Returns:
        Any: Result.
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    """Perform the print_numpy operation.

    Args:
        x (Tensor): Description.
        val (Any): Description.
        shp (Any): Description.

    Returns:
        Any: Result.
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    """Perform the mkdirs operation.

    Args:
        paths (str): Description.

    Returns:
        Any: Result.
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """Perform the mkdir operation.

    Args:
        path (str): Description.

    Returns:
        Any: Result.
    """
    if not os.path.exists(path):
        os.makedirs(path)
