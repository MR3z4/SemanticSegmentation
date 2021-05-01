import kornia
from torch import nn

from utils.train_options import generate_edge_tensor
from utils.utils import soft_argmax


class EdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, num_classes=20, weight=None):
        super(EdgeLoss, self).__init__()
        self.ignore_index = ignore_index
        # self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
        self.criterion = nn.BCELoss(weight=weight)
        # self.criterion = nn.MSELoss()
        self.num_classes = num_classes

    def forward(self, preds, target):
        preds_max = soft_argmax(preds)
        preds_max /= (preds_max + 1e-32)
        # preds_max = (preds_max > 0) * 1
        # preds_edge = generate_edge_tensor(preds_max)
        preds_edge = kornia.filters.sobel(preds_max.unsqueeze(1) * 1.7888, eps=1e-32).squeeze()
        loss = 0.1 * self.criterion(preds_edge, target) + 0.9 * self.criterion((1 - preds_edge), (1 - target))
        return loss
