# -*- coding: utf-8 -*
# @Time : 2023/9/4 10:45
# @Author : 杨坤林
# @File : PhaseLoss.py
# @Software : PyCharm
import torch
import torch.nn as nn


# 定义相位损失函数
class PhaseLoss(nn.Module):
    def __init__(self):
        super(PhaseLoss, self).__init__()

    def forward(self, predicted, target):
        # Compute the Fourier transform
        predicted_fft = torch.fft.fft2(predicted, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))

        # Extracting phase information
        predicted_phase = torch.angle(predicted_fft)
        target_phase = torch.angle(target_fft)

        # Normalize the phase information to the [-π, π] range
        predicted_phase = predicted_phase - 2 * torch.pi * torch.round(predicted_phase / (2 * torch.pi))
        target_phase = target_phase - 2 * torch.pi * torch.round(target_phase / (2 * torch.pi))

        # Calculating phase loss
        error = predicted_phase - target_phase
        phase_loss = torch.mean(error ** 2)

        return phase_loss


# # 示例用法
# if __name__ == '__main__':
#     # 创建两个示例特征图（大小相同）
#     predicted_feature_map = torch.randn(1, 1, 64, 64)  # 示例的预测特征图
#     target_feature_map = torch.randn(1, 1, 64, 64)  # 示例的目标特征图
#
#     # 初始化相位损失函数
#     phase_loss_fn = PhaseLoss()
#
#     # 计算相位损失
#     loss = phase_loss_fn(predicted_feature_map, target_feature_map)
#
#     print("相位损失:", loss.item())