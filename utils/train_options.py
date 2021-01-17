import torch
from torch.autograd import Variable
import random
import numpy as np
from utils.mixup import mixup_data, mixup_criterion


def get_input(images, labels, opts, device, cur_iter):
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

        return images, (labels_a, labels_b, lam)
    else:
        return images, labels


def calc_loss(criterion, outputs, labels, opts):
    if opts.use_mixup:
        labels_a, labels_b, lam = labels
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
    else:
        loss = criterion(outputs, labels)

    return loss
