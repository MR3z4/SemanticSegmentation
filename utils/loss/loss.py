import os
from importlib import import_module

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils.loss.edgeloss import EdgeLoss
from utils.loss.focal import FocalLoss
from utils.loss.kl_loss import KLDivergenceLoss
from utils.loss.lovasz_softmax import LovaszSoftmax
from utils.loss.scploss import SCPLoss, moving_average, to_one_hot
from utils.loss.rmi import RMILoss


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp=None):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = len([int(id) for id in args.gpu_ids.split(',')])
        self.num_classes = args.num_classes
        self.lovasz = LovaszSoftmax(ignore_index=255)
        self.kldiv = KLDivergenceLoss(ignore_index=255)
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss_type.replace(" ", "").split('+'):
            if len(loss.split('*')) == 1:
                weight = 1
                loss_type = loss
            elif len(loss.split('*')) == 2:
                weight, loss_type = loss.split('*')
            else:
                raise Exception("Wrong Loss Weight")
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'CE':
                loss_function = nn.CrossEntropyLoss(ignore_index=255, reduction='mean',
                                                    weight=torch.Tensor(args.loss_weights))
            elif loss_type == 'FL':
                loss_function = FocalLoss(ignore_index=255, size_average=True)
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'EL':
                loss_function = EdgeLoss()
            elif loss_type == 'RMI':
                loss_function = RMILoss(num_classes=args.num_classes,
                                        rmi_radius=3,
                                        rmi_pool_way=1,
                                        rmi_pool_size=2,
                                        rmi_pool_stride=2,
                                        loss_weight_lambda=0.5)
            elif loss_type == 'SCP':
                loss_function = SCPLoss(ignore_index=255, lambda_1=1, lambda_2=1, lambda_3=0.1, num_classes=20,
                                        weight=torch.Tensor(args.loss_weights))
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )
            else:
                raise Exception("Wrong Loss Type")

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = args.device
        self.loss_module.to(device)
        # if args.precision == 'half': self.loss_module.half()
        if device != 'cpu' and self.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        # if args.load != '.': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, pred, target, edges=None, soft_preds=None, soft_edges=None, cycle_n=None):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                if l['type'] == 'SCP':
                    loss = l['function'](pred, [target, edges, soft_preds, soft_edges], cycle_n)
                if l['type'] == 'EL':
                    if soft_edges is not None:
                        loss1 = 0.5 * l['function'](pred, edges)
                        soft_edge = moving_average(soft_edges, to_one_hot(edges, num_cls=self.num_classes),
                                                   1.0 / (cycle_n + 1.0))
                        # loss2 = 0.5 * self.kldiv(pred, soft_edge, edges)
                        loss2 = 0.5 * l['function'](pred, soft_edge)
                        loss = loss1 + loss2
                    else:
                        loss = l['function'](pred, edges)
                else:
                    if soft_preds is not None:
                        # loss1 = 0.5 * self.lovasz(pred, target)
                        soft_pred = moving_average(soft_preds, to_one_hot(target, num_cls=self.num_classes),
                                                   1.0 / (cycle_n + 1.0))
                        # loss2 = 0.5 * self.kldiv(pred, soft_pred, target)
                        loss1 = 0.5 * l['function'](pred, target)
                        loss2 = 0.5 * l['function'](pred, soft_pred)
                        loss = loss1 + loss2
                    else:
                        loss = l['function'](pred, target)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        self.losses = losses
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()
