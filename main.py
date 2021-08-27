import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils import data
from tqdm import tqdm

import network
import utils
from metrics import StreamSegMetrics
from utils import schp
from utils.configs import get_argparser, get_dataset
from utils.loss import Loss
from utils.optimizers import create_optimizer
from utils.train_options import get_input, calc_loss


def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            images = images[:, [2, 1, 0]]  # for backbone
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            outputs = outputs['preds']
            if 'ACE2P' in opts.model:
                preds = outputs[0][0].detach().max(dim=1)[1].cpu().numpy()
            elif 'edgev1' in opts.model:
                preds = outputs[0].detach().max(dim=1)[1].cpu().numpy()
            else:
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score


def main(criterion):
    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    pretrained_backbone = False if "ACE2P" in opts.model else True
    model = network.model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride,
                                          pretrained_backbone=pretrained_backbone, use_abn=opts.use_abn)
    if opts.use_schp:
        schp_model = network.model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride,
                                                   pretrained_backbone=pretrained_backbone, use_abn=opts.use_abn)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    model_params = [{'params': model.backbone.parameters(), 'lr': 0.01 * opts.lr},
                    {'params': model.classifier.parameters(), 'lr': opts.lr}, ]
    optimizer = create_optimizer(opts, model_params=model_params)
    # optimizer = torch.optim.SGD(params=[
    #     {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
    #     {'params': model.classifier.parameters(), 'lr': opts.lr},
    # ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_epochs": cur_epochs,
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    cycle_n = 0

    if opts.use_schp and opts.schp_ckpt is not None and os.path.isfile(opts.schp_ckpt):
        # TODO: there is a problem with this part.
        checkpoint = torch.load(opts.schp_ckpt, map_location=torch.device('cpu'))
        schp_model.load_state_dict(checkpoint["model_state"])
        print("SCHP Model restored from %s" % opts.schp_ckpt)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.use_schp:
            schp_model = nn.DataParallel(schp_model)
            schp_model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_epochs = checkpoint["cur_epochs"] - 1  # to start from the last epoch for schp
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)
        if opts.use_schp:
            schp_model = nn.DataParallel(schp_model)
            schp_model.to(device)

    # ==========   Train Loop   ==========#
    if opts.test_only:
        model.eval()
        val_score = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
        print(metrics.to_str(val_score))
        return
    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        criterion.start_log()
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            # images = images.to(device, dtype=torch.float32)
            # labels = labels.to(device, dtype=torch.long)
            images, labels = get_input(images, labels, opts, device, cur_itrs)
            if opts.use_mixup:
                images, main_images = images
            else:
                main_images = None
            images = images[:, [2, 1, 0]]  # for backbone
            optimizer.zero_grad()
            outputs = model(images)

            if opts.use_schp:
                # Online Self Correction Cycle with Label Refinement
                soft_labels = []
                if cycle_n >= 1:
                    with torch.no_grad():
                        if opts.use_mixup:
                            soft_preds = [schp_model(main_images[0]), schp_model(main_images[1])]
                            soft_edges = [None, None]
                        else:
                            soft_preds = schp_model(images)
                            soft_edges = None
                        if 'ACE2P' in opts.model:
                            soft_edges = soft_preds[1][-1]
                            soft_preds = soft_preds[0][-1]
                            # soft_parsing = []
                            # soft_edge = []
                            # for soft_pred in soft_preds:
                            #     soft_parsing.append(soft_pred[0][-1])
                            #     soft_edge.append(soft_pred[1][-1])
                            # soft_preds = torch.cat(soft_parsing, dim=0)
                            # soft_edges = torch.cat(soft_edge, dim=0)
                else:
                    if opts.use_mixup:
                        soft_preds = [None, None]
                        soft_edges = [None, None]
                    else:
                        soft_preds = None
                        soft_edges = None
                soft_labels.append(soft_preds)
                soft_labels.append(soft_edges)
                labels = [labels, soft_labels]

            # loss = criterion(outputs, labels)
            loss = calc_loss(criterion, outputs, labels, opts, cycle_n)
            loss.backward()
            optimizer.step()

            criterion.batch_step(len(images))
            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            sub_loss_text = ''
            for sub_loss, sub_prop in zip(criterion.losses, criterion.loss):
                if sub_prop['weight'] > 0:
                    sub_loss_text += f", {sub_prop['type']}: {sub_loss.item():.4f}"
            print(f"\rEpoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, Loss={np_loss:.4f}{sub_loss_text}", end='')

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print(f"\rEpoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, Loss={interval_loss:.4f} {criterion.display_loss().replace('][',', ')}")
                interval_loss = 0.0
                torch.cuda.empty_cache()

            if (cur_itrs) % opts.save_interval == 0 and (cur_itrs) % opts.val_interval != 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score = validate(opts=opts, model=model, loader=val_loader, device=device,
                                     metrics=metrics)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))
                    # save_ckpt('/content/drive/MyDrive/best_%s_%s_os%d.pth' %
                    #           (opts.model, opts.dataset, opts.output_stride))
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                criterion.end_log(len(train_loader))
                return

        # Self Correction Cycle with Model Aggregation
        if opts.use_schp:
            if (cur_epochs + 1) >= opts.schp_start and (cur_epochs + 1 - opts.schp_start) % opts.cycle_epochs == 0:
                print(f'\nSelf-correction cycle number {cycle_n}')

                schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))
                cycle_n += 1
                schp.bn_re_estimate(train_loader, schp_model)
                schp.save_schp_checkpoint({
                    'state_dict': schp_model.state_dict(),
                    'cycle_n': cycle_n,
                }, False, "checkpoints", filename=f'schp_{opts.model}_{opts.dataset}_cycle{cycle_n}_checkpoint.pth')
                # schp.save_schp_checkpoint({
                #     'state_dict': schp_model.state_dict(),
                #     'cycle_n': cycle_n,
                # }, False, '/content/drive/MyDrive/', filename=f'schp_{opts.model}_{opts.dataset}_checkpoint.pth')
        torch.cuda.empty_cache()
        criterion.end_log(len(train_loader))


if __name__ == '__main__':
    opts = get_argparser().parse_args(args=[])
    if 'ACE2P' in opts.model:
        opts.loss_type = 'SCP'
        opts.use_mixup = False
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opts.device = device
    print("Device: %s" % device)
    criterion = Loss(opts)
    main(criterion)
    criterion.plot_loss('/content/drive/MyDrive/', len(criterion.log))
