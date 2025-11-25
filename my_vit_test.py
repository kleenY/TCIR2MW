# -*- coding: utf-8 -*
# @Author : Kunlin Yang
# @File : my_test.py
# @Software : PyCharm
"""Evaluation runner for ViT‑based models: argument handling, experiment logging, checkpoint loading, and dataset iteration. Representative functions: build_summary_dict, create_exp_dir, set_lr, test2, my_normal."""

from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import math
from os.path import join
import random
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
import pandas as pd
from options.test_options import TestOptions
from models import create_model
from sklearn.metrics import mean_absolute_error

parser = argparse.ArgumentParser(description='my ir2mw test step')

parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
parser.add_argument('--modle_dir', type=str, default="")  # Path to pretrained weights
parser.add_argument('--my_train_logs_path', type=str, default="./my_train_logs")   # Directory to save logs
parser.add_argument('--dataset', type=str,
                    default='G:\\Inf2mw\\data\\val', help='data path')
parser.add_argument('--my_save_path', type=str,
                    default='./test', help='data path')

# Create dictionary
def build_summary_dict(total_losses, psnr, phase, summary_losses=None):

    if summary_losses is None:
        summary_losses = {}

    summary_losses["{}_loss_mean".format(phase)] = np.nanmean(total_losses)
    summary_losses["{}_psnr_mean".format(phase)] = np.nanmean(psnr)

    return summary_losses

# Create save directory
def create_exp_dir(exp):
    if not os.path.isdir(exp):  # Create if not exists
        os.makedirs(exp)

# Set up optimizer
def set_lr(args, epoch, optimizer):
    lrDecay = args.lrDecay
    decayType = args.decayType
    if decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2**epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'inv':
        k = 1 / lrDecay
        lr = args.lr / (1 + k * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



# 
# Purpose: Iterate over the evaluation set, run inference, and collect outputs/metrics.
def test2():
    args = parser.parse_args()

    opt = TestOptions().parse()



    create_exp_dir(args.my_train_logs_path)
    create_exp_dir(args.my_save_path)
    # 
    manualSeed = 101
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    print("Random Seed: ", manualSeed)

    device = torch.device("cuda:0" if args.cuda else "cpu")

    # 
    model = create_model(opt)


    df = pd.DataFrame(columns=['data_name', 'test Loss', 'test psnr'])  # 
    df.to_csv(os.path.join(args.my_train_logs_path, "test_summary.csv"), index=False)  # 


    image_dir = args.dataset

    lr_dir = join(image_dir, "b1")
    hr_dir = join(image_dir, "mw")

    image_names = os.listdir(lr_dir)
    i = 0

    psnr = []
    save_path = os.path.join(args.my_save_path, 'result')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for image_name in image_names:
        img = np.load(os.path.join(lr_dir, image_name))

        input_normal = img / 350
        input = torch.from_numpy(np.expand_dims(np.expand_dims(input_normal, 0), 0))
        input = input.to(device)

        target = np.load(os.path.join(hr_dir, image_name))
        # Normalization
        normal_target = target / 350
        target = torch.from_numpy(np.expand_dims(np.expand_dims(normal_target, 0), 0))
        target = target.to(device)


        model.set_input(input, target)
        with torch.no_grad():
            model.test()
            output = model.fake_B

        out_img = output.detach().squeeze().cpu().numpy()

        out_img = np.where(out_img > 1, 1, out_img)
        out_img = np.where(out_img < 0, 0, out_img)
        normal_target = np.where(normal_target > 1, 1, normal_target)
        normal_target = np.where(normal_target < 0, 0, normal_target)

        input_normal = np.where(input_normal > 1, 1, input_normal)
        input_normal = np.where(input_normal < 0, 0, input_normal)

        out_img_inv = out_img * 350

        temp_psnr = sk_psnr(normal_target, out_img)
        ssim = sk_ssim(normal_target.squeeze(), out_img.squeeze())
        mse = compare_mse(normal_target, out_img)
        rmse = mse ** 0.5
        # temp_lpips = lpips_vgg(torch.tensor(normal_target, dtype=torch.double), torch.tensor(out_img, dtype=torch.double))
        mae = mean_absolute_error(normal_target, out_img)

        psnr.append(temp_psnr)
        temp_save_path = os.path.join(save_path, image_name)
        np.save(temp_save_path, out_img_inv)

        psnr1 = "%f" % temp_psnr
        ssim1 = "%f" % ssim
        rmse1 = "%f" % rmse
        mae1 = "%f" % mae

        data_name = "%s" % image_name

        list = [data_name, psnr1, rmse1, ssim1, mae1]


        data = pd.DataFrame([list])

        data.to_csv(os.path.join(args.my_train_logs_path, "test_summary.csv"), mode='a', header=False,
                    index=False)  # modea,csv

    avg_psnr = np.mean(psnr)

    print(avg_psnr)


def my_normal(x):
    smax = np.max(x)
    smin = np.min(x)
    s = (x - smin)/(smax - smin)
    return s






if __name__ == '__main__':
    print("start!")
    test2()
    print("over!")