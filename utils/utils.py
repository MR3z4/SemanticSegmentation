import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import normalize


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean / std
    _std = 1 / std
    return normalize(tensor, _mean, _std)


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean / std
        self._std = 1 / std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)
        return normalize(tensor, self._mean, self._std)


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def add_void(label, width=3, void_value=255):
    mask = label.copy()
    edge = cv2.Canny(mask, 0, 0)
    edge = cv2.dilate(edge, np.ones((width, width)))
    mask[edge == 255] = void_value
    return mask


def soft_argmax(voxels):
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W)
    Return: 3D coordinates in shape (batch_size, H, W)
    """
    assert voxels.dim() == 4
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 10000.0
    N, C, H, W = voxels.shape
    soft_max = nn.functional.softmax(voxels.permute(0, 2, 3, 1) * alpha, dim=3)
    indices_kernel = torch.arange(start=0, end=C, dtype=torch.float32, device=voxels.device).unsqueeze(0)
    conv = soft_max * indices_kernel
    indices = conv.sum(3)
    out = indices % C

    return out
