import torch
import torch.nn.functional as F
from torch import nn


def flatten_probas(input, target, labels, ignore=255):
    """
    Flattens predictions in the batch.
    """
    B, C, H, W = input.size()
    input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    target = target.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return input, target
    valid = (labels != ignore)
    vinput = input[valid.nonzero().squeeze()]
    vtarget = target[valid.nonzero().squeeze()]
    return vinput, vtarget


class KLDivergenceLoss(nn.Module):
    def __init__(self, ignore_index=255, T=1):
        super(KLDivergenceLoss, self).__init__()
        self.ignore_index=ignore_index
        self.T = T

    def forward(self, input, target, label):
        log_input_prob = F.log_softmax(input / self.T, dim=1)
        target_porb = F.softmax(target / self.T, dim=1)
        loss = F.kl_div(*flatten_probas(log_input_prob, target_porb, label, ignore=self.ignore_index))
        return self.T*self.T*loss # balanced
