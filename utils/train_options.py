import torch
from networkx import edges
from torch.autograd import Variable
import random
import numpy as np
from torch.nn import functional as F
from utils.mixup import mixup_data, mixup_criterion


def get_input(images, labels, opts, device, cur_iter):
    if 'ACE2P' in opts.model:
        edges = generate_edge_tensor(labels)
        edges = edges.type(torch.cuda.LongTensor)
    elif 'edge' in opts.model:
        edges = labels[1]
        edges = edges.to(device, dtype=torch.float32)
        labels = labels[0]
    images = images.to(device, dtype=torch.float32)
    labels = labels.to(device, dtype=torch.long)

    if opts.use_mixup:
        if opts.use_mixup_mwh:
            stage1, stage2 = (np.array(opts.mwh_stages) * opts.total_itrs).astype(int)
            mask = random.random()
            if cur_iter >= stage2:
                # threshold = math.cos( math.pi * (epoch - 150) / ((200 - 150) * 2))
                threshold = (opts.total_itrs - cur_iter) / (opts.total_itrs - stage2)
                # threshold = 1.0 - math.cos( math.pi * (200 - epoch) / ((200 - 150) * 2))
                if mask < threshold:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, opts.mixup_alpha, device)
                else:
                    labels_a, labels_b = labels, labels
                    lam = 1.0
            elif cur_iter >= stage1:
                # in the main paper it was each epochs or mini batch, here i changed it to val_interval iterations
                if (cur_iter // opts.val_interval) % 2 == 0:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, opts.mixup_alpha, device)
                else:
                    labels_a, labels_b = labels, labels
                    lam = 1.0
            else:
                images, labels_a, labels_b, lam = mixup_data(images, labels, opts.mixup_alpha, device)
        else:
            images, labels_a, labels_b, lam = mixup_data(images, labels, opts.mixup_alpha, device)
            images, labels_a, labels_b = map(Variable, (images, labels_a, labels_b))

        return images, [labels_a, labels_b, lam]
    else:
        if 'ACE2P' or 'edge' in opts.model:
            return images, [labels, edges]
        else:
            return images, labels


def calc_loss(criterion, outputs, labels, opts):
    if opts.use_mixup:
        labels_a, labels_b, lam = labels
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
    else:
        if 'ACE2P' in opts.model:
            loss = criterion(outputs, labels[0], edges=labels[1])
        elif 'edge' in opts.model:
            loss_fusion = criterion(outputs[0], labels[0])
            loss_class = criterion(outputs[1], labels[0])
            loss_edge = torch.nn.MSELoss()(outputs[2], labels[1])
            loss = loss_class + loss_edge + loss_fusion
        else:
            loss = criterion(outputs, labels)

    return loss


def generate_edge_tensor(label, edge_width=3):
    label = label.type(torch.cuda.FloatTensor)
    if len(label.shape) == 2:
        label = label.unsqueeze(0)
    n, h, w = label.shape
    edge = torch.zeros(label.shape, dtype=torch.float).cuda()
    # right
    edge_right = edge[:, 1:h, :]
    edge_right[(label[:, 1:h, :] != label[:, :h - 1, :]) & (label[:, 1:h, :] != 255)
               & (label[:, :h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :, :w - 1]
    edge_up[(label[:, :, :w - 1] != label[:, :, 1:w])
            & (label[:, :, :w - 1] != 255)
            & (label[:, :, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:, :h - 1, :w - 1]
    edge_upright[(label[:, :h - 1, :w - 1] != label[:, 1:h, 1:w])
                 & (label[:, :h - 1, :w - 1] != 255)
                 & (label[:, 1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:, :h - 1, 1:w]
    edge_bottomright[(label[:, :h - 1, 1:w] != label[:, 1:h, :w - 1])
                     & (label[:, :h - 1, 1:w] != 255)
                     & (label[:, 1:h, :w - 1] != 255)] = 1

    kernel = torch.ones((1, 1, edge_width, edge_width), dtype=torch.float).cuda()
    with torch.no_grad():
        edge = edge.unsqueeze(1)
        edge = F.conv2d(edge, kernel, stride=1, padding=1)
    edge[edge != 0] = 1
    edge = edge.squeeze()
    return edge
