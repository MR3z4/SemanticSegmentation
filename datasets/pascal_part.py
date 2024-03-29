import os
import random
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from utils import add_void
from utils.ext_transforms import get_affine_transform


# from ext_transforms import get_affine_transform


class PascalPartSegmentation(data.Dataset):
    def __init__(self, root, split, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, flip_prob=0.5, transform=None, void_pixels=0, return_edge=False):
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = flip_prob
        self.transform = transform
        self.dataset = split
        self.void_pixels = void_pixels
        self.return_edge = return_edge

        list_path = os.path.join(self.root, self.dataset + '_id.txt')
        train_list = [i_id.strip() for i_id in open(list_path)]

        self.train_list = train_list
        self.number_samples = len(self.train_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        train_item = self.train_list[index]

        im_path = os.path.join(self.root, 'images', train_item + '.jpg')
        parsing_anno_path = os.path.join(self.root, 'labels', train_item + '.png')

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)[..., ::-1]
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test':
            # Get pose annotation
            # parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)
            parsing_anno = np.array(Image.open(parsing_anno_path))
            if self.dataset == 'train' or self.dataset == 'trainval':
                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

                if random.random() <= self.flip_prob:
                    im = im[:, ::-1, :]
                    parsing_anno = parsing_anno[:, ::-1]
                    person_center[0] = im.shape[1] - person_center[0] - 1
                    # right_idx = [15, 17, 19]
                    # left_idx = [14, 16, 18]
                    # for i in range(0, 3):
                    #     right_pos = np.where(parsing_anno == right_idx[i])
                    #     left_pos = np.where(parsing_anno == left_idx[i])
                    #     parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                    #     parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

        trans = get_affine_transform(person_center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        if self.return_edge:
            width = 10
            # label_edge = cv2.Canny(parsing_anno, 0, 0)  # all parts edge
            label_edge = cv2.Canny(((parsing_anno > 0) * 1).astype('uint8'), 0, 0)  # whole human edge
            label_edge = cv2.dilate(label_edge, np.ones((width, width)))
            input_edge = cv2.Canny(input, 100, 200, L2gradient=True)

            edge_parsing = cv2.warpAffine(
                label_edge,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0)

            edge_parsing = (edge_parsing.astype(int) * input_edge.astype(int)).clip(max=255).astype('uint8')
            edge_parsing = cv2.dilate(edge_parsing, np.ones((3, 3))).clip(max=1)

            edge_parsing = torch.from_numpy(edge_parsing)

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': train_item,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset == 'test':
            return input, meta
        else:

            if self.void_pixels > 0:
                parsing_anno = add_void(parsing_anno, width=self.void_pixels, void_value=self.ignore_label)
            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(self.ignore_label))

            label_parsing = torch.from_numpy(label_parsing)

            if self.return_edge:
                return input, (label_parsing, edge_parsing)

            return input, label_parsing


class PascalPartValSegmentation(data.Dataset):
    def __init__(self, root, crop_size=[473, 473], ignore_label=255, transform=None, ext='.jpg'):
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.transform = transform

        list_path = os.path.join(self.root, f'*{ext}')
        train_list = glob.glob(list_path)

        self.train_list = train_list
        self.number_samples = len(self.train_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        train_item = self.train_list[index]

        im_path = os.path.join(train_item)
        # parsing_anno_path = os.path.join(self.root, 'labels', train_item + '.png')

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)[..., ::-1]
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        trans = get_affine_transform(person_center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': train_item,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }
        #
        # label_parsing = cv2.warpAffine(
        #     parsing_anno,
        #     trans,
        #     (int(self.crop_size[1]), int(self.crop_size[0])),
        #     flags=cv2.INTER_NEAREST,
        #     borderMode=cv2.BORDER_CONSTANT,
        #     borderValue=(255))

        # label_parsing = torch.from_numpy(label_parsing)

        return input, meta

    def transform_logits(self, logits, center, scale, width, height, input_size):

        trans = get_affine_transform(center, scale, 0, input_size, inv=1)
        channel = logits.shape[2]
        target_logits = []
        for i in range(channel):
            target_logit = cv2.warpAffine(
                logits[:, :, i],
                trans,
                (int(width), int(height)),  # (int(width), int(height)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0))
            target_logits.append(target_logit)
        target_logits = np.stack(target_logits, axis=2)

        return target_logits


if __name__ == '__main__':
    from tqdm import tqdm

    num_classes = 7
    train_dst = PascalPartSegmentation(root='./data/pascalpart', split='train', crop_size=[512, 512], scale_factor=0,
                                       rotation_factor=0, ignore_label=255, flip_prob=0, transform=None)
    train_loader = data.DataLoader(
        train_dst, batch_size=32, shuffle=False, num_workers=2)
    weights = np.zeros(num_classes)
    for (images, labels) in tqdm(train_loader):
        labels = labels.numpy()
        pixel_nums = []
        tot_pixels = 0
        for i in range(num_classes):
            pixel_num_of_cls_i = np.sum(labels == i).astype(np.float)
            pixel_nums.append(pixel_num_of_cls_i)
            tot_pixels += pixel_num_of_cls_i
        weight = []
        for i in range(num_classes):
            weight.append(
                (tot_pixels - pixel_nums[i]) / tot_pixels / (num_classes - 1)
            )
        weight = np.array(weight, dtype=np.float)
        weights += weight
        # weights = torch.from_numpy(weights).float().to(masks.device)
    print(weights / len(train_loader))
