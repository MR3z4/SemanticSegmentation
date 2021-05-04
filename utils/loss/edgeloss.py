import kornia
from torch import nn
from torch.nn.functional import binary_cross_entropy
import torch
from utils.train_options import generate_edge_tensor
from utils.utils import soft_argmax


class EdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, num_classes=20, weight=None):
        super(EdgeLoss, self).__init__()
        self.ignore_index = ignore_index
        # self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
        # self.criterion = nn.BCELoss(weight=weight)
        self.criterion = WeightedMSELoss()
        # self.criterion = nn.MSELoss()
        self.num_classes = num_classes

    def forward(self, pred, target):
        pred_max = soft_argmax(pred)
        pred_max /= (pred_max + 1e-32)
        # preds_max = (preds_max > 0) * 1
        # preds_edge = generate_edge_tensor(preds_max)
        pred_edge = kornia.filters.sobel(pred_max.unsqueeze(1) * 1.7888, eps=1e-32).squeeze()
        weight = target + 0.1
        loss = self.criterion(pred_edge, target, weight=weight)
        return loss


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, pred, target, weight=None):
        diff2 = (torch.flatten(pred) - torch.flatten(target)) ** 2.0
        if weight is None:
            weight = torch.ones_like(target)
        flat_weight = torch.flatten(weight)
        assert(len(flat_weight) == len(diff2))
        loss = (diff2 * flat_weight).sum() / flat_weight.sum()
        return loss