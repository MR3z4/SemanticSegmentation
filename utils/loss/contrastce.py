import torch
from torch import nn
import torch.nn.functional as F

from utils.loss.rmi import RMILoss


# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, ce_weight=None, ce_reduction='elementwise_mean', ignore_index=255):  # , configer=None):
        super(FSCELoss, self).__init__()
        # self.configer = configer
        weight = ce_weight
        # weight = None
        # if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
        #     weight = self.configer.get('loss', 'params')['ce_weight']
        #     weight = torch.FloatTensor(weight).cuda()

        reduction = ce_reduction
        # reduction = 'elementwise_mean'
        # if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
        #     reduction = self.configer.get('loss', 'params')['ce_reduction']

        # ignore_index = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class FSRMILoss(nn.Module):
    def __init__(self, num_classes, seg_loss_weight=1.0, ignore_index=255):  # , configer=None):
        super(FSRMILoss, self).__init__()
        # self.configer = configer

        # num_classes = self.configer.get('data', 'num_classes')

        # ignore_index = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']

        self.seg_loss_weight = seg_loss_weight
        self.rmi_loss = RMILoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, inputs, targets, **kwargs):
        # if isinstance(inputs, dict):
        #     seg_out = inputs
        # else:
        #     aux_out, seg_out = inputs
        seg_out = inputs
        seg_loss = self.rmi_loss(seg_out, targets)
        # loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        # loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
        loss = self.seg_loss_weight * seg_loss
        return loss


class PixelContrastLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, max_samples=1024, max_views=100,
                 ignore_label=255):  # , configer):
        super(PixelContrastLoss, self).__init__()

        # self.configer = configer
        # self.temperature = self.configer.get('contrast', 'temperature')
        # self.base_temperature = self.configer.get('contrast', 'base_temperature')

        self.temperature = temperature
        self.base_temperature = base_temperature

        self.ignore_label = ignore_label
        # self.ignore_label = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

        # self.max_samples = self.configer.get('contrast', 'max_samples')
        # self.max_views = self.configer.get('contrast', 'max_views')
        self.max_samples = max_samples
        self.max_views = max_views

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x > 0 and x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    # Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        epsilon = 1e-36
        log_prob = logits - torch.log(exp_logits + neg_logits + epsilon)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        predict = predict.unsqueeze(1).float().clone()
        predict = torch.nn.functional.interpolate(predict,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        if feats_ is None:
            return 0
        loss = self._contrastive(feats_, labels_)

        return loss


class ContrastCELoss(nn.Module):
    def __init__(self, num_classes, loss_weight=0.1, seg_loss_weight=1.0, use_rmi=True, temperature=0.07,
                 base_temperature=0.07, max_samples=1024, max_views=30, ce_weight=None,
                 ce_reduction="elementwise_mean", ignore_index=255):  # configer=None):
        super(ContrastCELoss, self).__init__()

        # self.configer = configer

        # ignore_index = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        # Log.info('ignore_index: {}'.format(ignore_index))

        # self.loss_weight = self.configer.get('contrast', 'loss_weight')
        # self.use_rmi = self.configer.get('contrast', 'use_rmi')

        self.loss_weight = loss_weight
        self.use_rmi = use_rmi

        if self.use_rmi:
            self.seg_criterion = FSRMILoss(num_classes, seg_loss_weight=seg_loss_weight, ignore_index=ignore_index)
        else:
            self.seg_criterion = FSCELoss(seg_loss_weight=seg_loss_weight, ce_weight=ce_weight,
                                          ce_reduction=ce_reduction, ignore_index=ignore_index)

        self.contrast_criterion = PixelContrastLoss(temperature=temperature, base_temperature=base_temperature,
                                                    max_samples=max_samples, max_views=max_views,
                                                    ignore_label=ignore_index)

    def forward(self, preds, target, embedding=None):
        h, w = target.size(1), target.size(2)

        # seg = preds['seg']
        # seg_aux = preds['seg_aux']
        seg = preds['preds']
        embedding = preds['features']['out'] if 'features' in preds else None

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)

        if embedding is not None:
            _, predict = torch.max(seg, 1)

            loss_contrast = self.contrast_criterion(embedding, target, predict)
            return loss + self.loss_weight * loss_contrast

        return loss
