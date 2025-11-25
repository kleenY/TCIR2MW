# -*- coding: utf-8 -*
# @Time : 2023/9/4 10:45
# @Author : Kunlin Yang
# @File : PhaseLoss.py
# @Software : PyCharm
"""Phase‑aware loss defined in the frequency domain to penalize phase inconsistency between predictions and targets. Prominent classes: PhaseLoss."""

import torch
import torch.nn as nn


# phase loss
class PhaseLoss(nn.Module):
    def __init__(self):
        super(PhaseLoss, self).__init__()

    # Purpose: Compute the phase‑domain discrepancy and return a scalar loss.
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
        phase_loss = torch.mean(torch.abs(predicted_phase - target_phase))

        return phase_loss


# # 
# if __name__ == '__main__':
#     # ()
#     predicted_feature_map = torch.randn(1, 1, 64, 64)  # 
#     target_feature_map = torch.randn(1, 1, 64, 64)  # 
#
#     # phase loss
#     phase_loss_fn = PhaseLoss()
#
#     # phase loss
#     loss = phase_loss_fn(predicted_feature_map, target_feature_map)
#
#     print("phase loss:", loss.item())