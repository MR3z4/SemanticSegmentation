import numpy as np
import torch


def mixup_data(x, y, alpha=1.0, device='cuda', has_edge=False):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    if has_edge:
        y_a = y
        y_b = [y[0][index], y[1][index]]
    else:
        y_a, y_b = y, y[index]

    return [mixed_x, [x, x[index]]], y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, edges=False, soft_preds=None, soft_edges=None, cycle_n=None):
    if edges:
        return lam * criterion(pred, y_a[0], edges=y_a[1], soft_preds=soft_preds[0], soft_edges=soft_edges[0],
                               cycle_n=cycle_n) + (1 - lam) * criterion(pred, y_b[0], edges=y_b[1],
                                                                        soft_preds=soft_preds[1],
                                                                        soft_edges=soft_edges[1], cycle_n=cycle_n)
    else:
        return lam * criterion(pred, y_a, edges=edges, soft_preds=soft_preds[0], soft_edges=soft_edges[0],
                               cycle_n=cycle_n) + (1 - lam) * criterion(pred, y_b, edges=edges,
                                                                        soft_preds=soft_preds[1],
                                                                        soft_edges=soft_edges[1], cycle_n=cycle_n)
