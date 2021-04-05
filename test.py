import cv2
from torch.utils import data
from tqdm import tqdm
import network
from datasets import PascalPartValSegmentation
from utils import ext_transforms as et, utils, add_void
from torchvision.transforms import transforms
import torch
import numpy as np
import matplotlib
# matplotlib.use('Agg')

from PIL import Image
import matplotlib.pyplot as plt

from utils.ext_transforms import get_affine_transform

num_classes = 7
output_stride = 16
device = 'cuda'
model_name = 'deeplabv3plus_resnet101v2'
# model_name = 'ACE2P_resnet101'
checkpoint_path = 'best_deeplabv3plus_resnet101_pascalpart_os16_ce_6669_mixwh_ms.pth'
# checkpoint_path = r"ace2p_initial_abn.pth"


# img_path = 'samples/23_image.png'
# lbl_path = 'samples/1_target.png'

model_map = network.model_map

model = model_map[model_name](num_classes=num_classes, output_stride=output_stride, pretrained_backbone=True,
                              use_abn=False)
chk = torch.load(checkpoint_path)
model.load_state_dict(chk['model_state'])
# model.load_state_dict(chk)

model.to(device)
model.eval()
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
])
denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])

val_dst = PascalPartValSegmentation(root="testimages", ext='.jpg', crop_size=[512, 512], ignore_label=255,
                                    transform=val_transform)
val_loader = data.DataLoader(val_dst, batch_size=1, shuffle=False, num_workers=0)

for (image, meta) in tqdm(val_loader):
    with torch.no_grad():
        images = image[:, [2, 1, 0]].to(device, dtype=torch.float32)
        metas = meta

        outputs = model(images)
        if 'ACE2P' in model_name:
            preds = outputs[0][1].detach().max(dim=1)[1].cpu().numpy()[0]
        else:
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()[0]
        img = (denorm(images[0].detach().cpu().numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
        # targets = labels.cpu().numpy()

        gt = Image.open(
            r"C:\Users\MohammadReza\Desktop\Thesis\Self-Correction-Human-Parsing-Results\gt\IMG_20201114_124215_987.png")
        gt_ = np.array(gt)
        gt_[(gt_ == (128, 0, 0)).all(2)] = 1
        gt_[(gt_ == (0, 128, 0)).all(2)] = 2
        gt_[(gt_ == (128, 128, 0)).all(2)] = 3
        gt_[(gt_ == (0, 0, 128)).all(2)] = 4
        gt_[(gt_ == (128, 0, 128)).all(2)] = 5
        gt_[(gt_ == (0, 128, 128)).all(2)] = 6
        h, w, _ = gt_.shape
        person_center, s = PascalPartValSegmentation._box2cs(val_dst, [0, 0, w - 1, h - 1])
        r = 0

        trans = get_affine_transform(person_center, s, r, val_dst.crop_size)
        gt_ = cv2.warpAffine(
            gt_,
            trans,
            (int(val_dst.crop_size[1]), int(val_dst.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        from metrics import StreamSegMetrics

        met = StreamSegMetrics(7)
        met.update(np.expand_dims(gt_[..., 0], 0), np.expand_dims(preds, 0))
        print(met.get_results())
        plt.imshow(preds)
        plt.show()
