# -*- coding: utf-8 -*
# @Time : 2022/5/6 22:46
# Author: Kunlin Yang
# @File : numpy_to_image.py
# @Software : PyCharm

import os
import random
import time

import cv2 as cv
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np

def draw(data_path):
    '''
    单独的画图函数，将npy文件保存到指定路径
    '''
    data = np.load(data_path)
    data = data.squeeze()
    plt.figure()
    sns.heatmap(data=data,
                cmap="gray_r"  # Pastel palette (sns.light_palette) usage：sns.light_palette()
                )
    plt.title(os.path.basename(data_path))
    # plt.show()
    plt.savefig("new_test.png")
    plt.close()

def convert1(path_test):
    """Perform the convert1 operation.

    Args:
        path_test (str): Description.

    Returns:
        Any: Result.
    """
    # image_arrayNormalization
    image_array = np.load(path_test)
    # Normalize to [0, 1]，,，Normalization
    arr_no_0 = image_array.flatten()[np.flatnonzero(image_array)]
    max = arr_no_0.max()
    min = arr_no_0.min()
    # Comment translated to English (manual check recommended)
    image_array = np.where(image_array == 0, max, image_array)
    image_array = (image_array - min) / (max - min)
    # (image_array-np.min(image_array))/(np.max(image_array)-np.min(image_array))  # Normalization

    image_array = 1 - image_array
    image_array *= 255  # Convert to 0–255 grayscale values
    # image_array = 255 - image_array
    # print(image_array[255, 255])
    im = Image.fromarray(image_array)
    im1 = im.convert('F')  # Convert to grayscale; for color images use "RGB" instead of "L".
    print(im1.getpixel((255, 255)))
    print(type(im1.getpixel((255, 255))))
    im2 = im.convert('L')
    print(im2.getpixel((255, 255)))
    print(type(im2.getpixel((255, 255))))
    im3 = im.convert('I')  # Convert to grayscale; for color images use "RGB" instead of "L".
    print(im3.getpixel((255, 255)))
    print(type(im3.getpixel((255, 255))))
    # print(im.mode)
    im1.save('test1.tiff')
    im2.save('test2.png')
    im3.save('test3.png')

def convert2():
    """Perform the convert2 operation.

    Args:
        None

    Returns:
        Any: Result.
    """
    # Make an array of 120000 random bytes
    randomByteArray = bytearray(os.urandom(120000))
    # translate into numpy array
    flatNumpyArray = np.array(randomByteArray)
    # Convert the array to make a 400*300 grayscale image(grayscale image)
    grayImage = flatNumpyArray.reshape(300, 400)
    # show gray image
    cv.imshow('GrayImage', grayImage)
    # print image's array
    print(grayImage)
    cv.waitKey()

    # byte array translate into RGB image
    randomByteArray1 = bytearray(os.urandom(360000))
    flatNumpyArray1 = np.array(randomByteArray1)
    BGRimage = flatNumpyArray1.reshape(300, 400, 3)
    cv.imshow('BGRimage', BGRimage)
    cv.waitKey()
    cv.destroyAllWindows()

# Progress bar usage example
def pbar():
    """Perform the pbar operation.

    Args:
        None

    Returns:
        Any: Result.
    """
    pbar = tqdm(total=100)
    for i in range(100):
        time.sleep(1)
        pbar.update(1)

def convert3(path, savepath):
    """Perform the convert3 operation.

    Args:
        path (str): Description.
        savepath (str): Description.

    Returns:
        Any: Result.
    """
    # image_arrayNormalization
    image_array = np.load(path)
    # Normalize to [0, 1]，,，Normalization
    # arr_no_0 = image_array.flatten()[np.flatnonzero(image_array)]
    # max = np.max(image_array)
    # min = np.min(image_array)
    # Comment translated to English (manual check recommended)
    # image_array = np.where(image_array == 0, max, image_array)
    # Normalization
    image_array = (image_array - 100) / (350 - 100)
    # (image_array-np.min(image_array))/(np.max(image_array)-np.min(image_array))  # Normalization
    # image_array = np.where(image_array == 1, 0, image_array)
    # Inverse / reverse
    image_array = 1 - image_array

    # Convert to 0–255 grayscale values
    image_array *= 255
    # image_array = 255 - image_array
    # print(image_array[255, 255])
    im = Image.fromarray(image_array)
    # im1 = im.convert('F')  # Convert to grayscale; for color images use "RGB" instead of "L".
    # print(im1.getpixel((255, 255)))
    # print(type(im1.getpixel((255, 255))))
    im2 = im.convert('L')
    name = os.path.basename(path)
    name = name[:-4]
    name = name + '.png'
    temp_savepath = os.path.join(savepath, name)
    # print(im2.getpixel((255, 255)))
    # print(type(im2.getpixel((255, 255))))  # Difference between "F" and "L" modes: "L" stores integers.
    # im3 = im.convert('I')  # Convert to grayscale; for color images use "RGB" instead of "L".
    # print(im3.getpixel((255, 255)))
    # print(type(im3.getpixel((255, 255))))
    # print(im.mode)
    # im1.save('test1.png')
    im2.save(temp_savepath)
    # im3.save('test3.png')

def convert_all(npy_path, img_path):
    """Perform the convert_all operation.

    Args:
        npy_path (str): Description.
        img_path (Tensor): Description.

    Returns:
        Any: Result.
    """
    data_list = os.listdir(npy_path)
    # pbar = tqdm(total=len(data_list))
    for data in data_list:
        temp_path = os.path.join(npy_path, data)
        convert3(temp_path, img_path)
        # pbar.update(1)
    # pbar.close()

if __name__ == '__main__':
    print("start!")
    # pathtest = 'G:/meta_data/train/CH3_TEMP_IRSPL/lr/0.npy'
    # save_path = './'
    # convert3(pathtest, save_path)
    # # draw(pathtest)

    inf2mw_path = 'G:\\trn\\b1'
    image_inf_mw_path = 'G:\\trn\\b1_image'
    if not os.path.isdir(image_inf_mw_path):  # Create it if it does not exist
        os.makedirs(image_inf_mw_path)
    convert_all(inf2mw_path, image_inf_mw_path)

    # print("end!")

    # path_all = ''
    # model_list = os.listdir(path_all)
    # pbar = tqdm(total=len(model_list))
    # for model_name in model_list:
    #     temp_model_path = os.path.join(path_all, model_name)
    #     temp_npy_path = os.path.join(temp_model_path, 'npy')
    #     temp_img_path = os.path.join(temp_model_path, 'image')
    #     task_list = os.listdir(temp_npy_path)
    #     for task_name in task_list:
    #         npy_path = os.path.join(temp_npy_path, task_name)
    #         save_path = os.path.join(temp_img_path, task_name)
    # if not os.path.isdir(save_path):  # Create it if it does not exist
    #             os.makedirs(save_path)
    #         convert_all(npy_path, save_path)
    #
    #     pbar.update(1)
    # pbar.close()

    # npy_path = '../my_test/CH5_TEMP_IRWVP_npy'
    # save_path = '../my_test/CH5_TEMP_IRWVP_img'
    # if not os.path.isdir(save_path):  # Create it if it does not exist
    #     os.makedirs(save_path)
    # convert_all(npy_path, save_path)
    print("over!")
    # dataname = os.listdir("/media/aita-ocean/data/YKL/meta_train_data/test/CH3_TEMP_IRSPL/lr")
    # print(random.sample(dataname, 50))
