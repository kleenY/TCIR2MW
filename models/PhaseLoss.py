# -*- coding: utf-8 -*
# @Time : 2023/9/4 10:45
# Author: Kunlin Yang
# @File : PhaseLoss.py
# @Software : PyCharm
import torch
import torch.nn as nn

# phase loss
class PhaseLoss(nn.Module):
    """Class PhaseLoss.

    Notes:
        Auto-generated documentation. Please refine as needed.
    """
    def __init__(self):
        """Initialize the instance.

        Args:
            None

        Returns:
            Any: Result.
        """
        super(PhaseLoss, self).__init__()

    def forward(self, predicted, target):
        """Run the forward pass of the network.

        Args:
            predicted (Any): Description.
            target (Any): Description.

        Returns:
            Tensor: Result.
        """
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

# Comment translated to English (manual check recommended)
# if __name__ == '__main__':
# Comment translated to English (manual check recommended)
# Comment translated to English (manual check recommended)
# Comment translated to English (manual check recommended)
#
# # phase loss
#     phase_loss_fn = PhaseLoss()
#
# # phase loss
#     loss = phase_loss_fn(predicted_feature_map, target_feature_map)
#
# print("phase loss:", loss.item())
