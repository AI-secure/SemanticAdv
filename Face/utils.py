import functools as ft
import operator as op
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def rec_transform(imgs):
    return (imgs - 0.5) * 4


'''reduce_* helper functions reduce tensors on all dimensions but the first.
They are intended to be used on batched tensors where dim 0 is the batch dim.
'''


def save_image(save_path, img, flag=2):
    path = "/".join(save_path.split('/')[:-1])
    if not osp.exists(path):
        os.makedirs(path)
    img = img.data.cpu().numpy()[0]
    if flag == 0:  #(-1, 1 )
        img = (img + 1) * 0.5
    elif flag == 1:  #(-2,2)
        img = (img + 2) * 0.25
    else:
        pass
    img = img * 255.0
    img = np.array(img).astype(np.uint8).transpose(1, 2, 0)
    img_pil = Image.fromarray(img)
    img_pil.save(save_path)


def save_npy(save_path, npy):
    path = "/".join(save_path.split('/')[:-1])
    if not osp.exists(path):
        os.makedirs(path)
    npy = npy.data.cpu().numpy()[0]
    np.save(save_path, npy)


def save_npz(save_path, npz):
    path = "/".join(save_path.split('/')[:-1])
    if not osp.exists(path):
        os.makedirs(path)
    npz = npz.data.cpu().numpy()[0]
    np.savez(save_path, npz)


def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x


def reduce_mean(x, keepdim=True):
    numel = ft.reduce(op.mul, x.size()[1:])
    x = reduce_sum(x, keepdim=keepdim)
    return x / numel


def reduce_min(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.min(a, keepdim=keepdim)[0]
    return x


def reduce_max(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.max(a, keepdim=keepdim)[0]
    return x


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


def l2r_dist(x, y, keepdim=True, eps=1e-8):
    d = (x - y)**2
    d = reduce_sum(d, keepdim=keepdim)
    d += eps  # to prevent infinite gradient at 0
    return d.sqrt()


def l2_dist(x, y, keepdim=True):
    d = (x - y)**2
    return reduce_sum(d, keepdim=keepdim)


def l1_dist(x, y, keepdim=True):
    d = torch.abs(x - y)
    return reduce_sum(d, keepdim=keepdim)


def l2_norm(x, keepdim=True):
    norm = reduce_sum(x * x, keepdim=keepdim)
    return norm.sqrt()


def l1_norm(x, keepdim=True):
    return reduce_sum(x.abs(), keepdim=keepdim)


def rescale(x, x_min=-1., x_max=1.):
    return x * (x_max - x_min) + x_min


def tanh_rescale(x, x_min=-1., x_max=1.):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1, size_average=True, L=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.size_average = size_average
        self.L = L

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        if self.L == 1:
            h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
            w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        else:
            h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
            w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        if not self.size_average:
            return self.TVLoss_weight * (h_tv + w_tv)
        else:
            count_h = self._tensor_size(x[:, :, 1:, :])
            count_w = self._tensor_size(x[:, :, :, 1:])
            return self.TVLoss_weight * 2 * (h_tv / count_h +
                                             w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
