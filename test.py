from torch.utils import data
from tqdm import tqdm
import network
from datasets import PascalPartValSegmentation
from utils import ext_transforms as et, utils
from torchvision.transforms import transforms
import torch
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

num_classes = 7
output_stride = 16
device = 'cuda'
model_name = 'deeplabv3plus_resnet101'
checkpoint_path = 'best_deeplabv3plus_mobilenet_voc_os16.pth'
img_path = 'samples/23_image.png'
lbl_path = 'samples/1_target.png'

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
# model.load_state_dict(chk['model_state'])
model.to(device)
model.eval()
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])

val_dst = PascalPartValSegmentation(root="samples", ext='.jpg', crop_size=[512, 512], ignore_label=255, transform=val_transform)
val_loader = data.DataLoader(val_dst, batch_size=1, shuffle=False, num_workers=0)

for (image, meta) in tqdm(val_loader):
    with torch.no_grad():
        images = image.to(device, dtype=torch.float32)
        metas = meta

        outputs = model(images)
        preds = outputs.detach().max(dim=1)[1].cpu().numpy()[0]
        img = (denorm(images[0].detach().cpu().numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
        # targets = labels.cpu().numpy()
        plt.imshow(preds)
        plt.show()