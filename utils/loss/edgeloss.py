from torch import nn

from utils.train_options import generate_edge_tensor
from utils.utils import soft_argmax

class EdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, num_classes=20, weight=None):
        super(EdgeLoss, self).__init__()
        self.ignore_index = ignore_index
        # self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
        self.criterion = nn.MSELoss()
        self.num_classes = num_classes
    def forward(self, preds, target):
        preds_max = soft_argmax(preds)
        preds_edge = generate_edge_tensor(preds_max)
        loss = self.criterion(preds_edge, target)
        return loss