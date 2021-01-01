from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from datasets.voc import voc_cmap
from utils import ext_transforms as et

import torch

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

num_classes = 21
output_stride = 16
device = 'cpu'
model_name = 'deeplabv3plus_mobilenet'
checkpoint_path = 'best_deeplabv3plus_mobilenet_voc_os16.pth'
img_path = 'samples/23_image.png'
lbl_path = 'samples/1_target.png'
cmap = voc_cmap()

model_map = {
    'deeplabv3_resnet50': network.deeplabv3_resnet50,
    'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
    'deeplabv3_resnet101': network.deeplabv3_resnet101,
    'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
    'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
}

model = model_map[model_name](num_classes=num_classes, output_stride=output_stride, pretrained_backbone=False)
chk = torch.load(checkpoint_path)
model.load_state_dict(chk['model_state'])
model.to(device)
model.eval()
# val_transform = et.ExtCompose([
#     et.ExtToTensor(),
#     et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
# ])
val_transform = et.ExtCompose([
    et.ExtResize(513),
    et.ExtCenterCrop(513),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

img = Image.open(img_path)
lbl = Image.open(lbl_path)
image, label = val_transform(img, lbl)

with torch.no_grad():
    images = image.unsqueeze(0).to(device, dtype=torch.float32)
    labels = label.unsqueeze(0).to(device, dtype=torch.long)

    outputs = model(images)
    preds = outputs.detach().max(dim=1)[1].cpu().numpy()
    targets = labels.cpu().numpy()
    plt.imshow(preds[0])
    plt.show()