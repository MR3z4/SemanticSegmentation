import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from .lovasz_softmax import LovaszSoftmax
from .kl_loss import KLDivergenceLoss
from .consistency_loss import ConsistencyLoss

NUM_CLASSES = 20


class SCPLoss(nn.Module):
    def __init__(self, ignore_index=255, lambda_1=1, lambda_2=1, lambda_3=1,
                 num_classes=20, weight=None):
        super(SCPLoss, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
        self.lovasz = LovaszSoftmax(ignore_index=ignore_index)
        self.kldiv = KLDivergenceLoss(ignore_index=ignore_index)
        self.reg = ConsistencyLoss(ignore_index=ignore_index)
        self.lamda_1 = lambda_1
        self.lamda_2 = lambda_2
        self.lamda_3 = lambda_3
        self.num_classes = num_classes

    def parsing_loss(self, preds, target, cycle_n=None):
        """
        Loss function definition.

        Args:
            preds: [[parsing result1, parsing result2],[edge result]]
            target: [parsing label, egde label]
            soft_preds: [[parsing result1, parsing result2],[edge result]]
        Returns:
            Calculated Loss.
        """
        h, w = target[0].size(1), target[0].size(2)

        pos_num = torch.sum(target[1] == 1, dtype=torch.float)
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])  # edge loss weight

        loss = 0

        # loss for segmentation
        preds_parsing = preds[0]
        for pred_parsing in preds_parsing:
            # scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
            #                            mode='bilinear', align_corners=True)

            loss += 0.5 * self.lamda_1 * self.lovasz(pred_parsing, target[0])
            if target[2] is None:
                loss += 0.5 * self.lamda_1 * self.criterion(pred_parsing, target[0])
            else:
                # soft_scale_pred = F.interpolate(input=target[2], size=(h, w),
                #                                 mode='bilinear', align_corners=True)
                soft_pred = moving_average(target[2], to_one_hot(target[0], num_cls=self.num_classes),
                                                 1.0 / (cycle_n + 1.0))
                loss += 0.5 * self.lamda_1 * self.kldiv(pred_parsing, soft_pred, target[0])

        # loss for edge
        preds_edge = preds[1]
        for pred_edge in preds_edge:
            # scale_pred = F.interpolate(input=pred_edge, size=(h, w),
            #                            mode='bilinear', align_corners=True)
            if target[3] is None:
                loss += self.lamda_2 * F.cross_entropy(pred_edge, target[1],
                                                       weights.cuda(), ignore_index=self.ignore_index)
            else:
                # soft_scale_edge = F.interpolate(input=target[3], size=(h, w),
                #                                 mode='bilinear', align_corners=True)
                soft_edge = moving_average(target[3], to_one_hot(target[1], num_cls=2),
                                                 1.0 / (cycle_n + 1.0))
                loss += self.lamda_2 * self.kldiv(pred_edge, soft_edge, target[0])

        # consistency regularization
        preds_parsing = preds[0]
        preds_edge = preds[1]
        for pred_parsing in preds_parsing:
            # scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
            #                            mode='bilinear', align_corners=True)
            # scale_edge = F.interpolate(input=preds_edge[0], size=(h, w),
            #                            mode='bilinear', align_corners=True)
            loss += self.lamda_3 * self.reg(pred_parsing, preds_edge[0], target[0])

        return loss

    def forward(self, preds, target, cycle_n=None):
        loss = self.parsing_loss(preds, target, cycle_n)
        return loss


def moving_average(target1, target2, alpha=1.0):
    target = 0
    target += (1.0 - alpha) * target1
    target += target2 * alpha
    return target


def to_one_hot(tensor, num_cls, dim=1, ignore_index=255):
    b, h, w = tensor.shape
    tensor[tensor == ignore_index] = 0
    onehot_tensor = torch.zeros(b, num_cls, h, w).cuda()
    onehot_tensor.scatter_(dim, tensor.unsqueeze(dim), 1)
    return onehot_tensor
