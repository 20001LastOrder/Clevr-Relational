import os
import json

import numpy as np
import cv2
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pycocotools.mask as mask_util
import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ClevrObjectDataset(Dataset):
    def __init__(self, obj_ann_path, img_dir, attr_names, min_img_id=None, max_img_id=None):
        with open(obj_ann_path) as f:
            anns = json.load(f)['objects']

        min_id = 0
        if min_img_id is not None:
            while anns['image_idxs'][min_id] < min_img_id:
                min_id += 1
        max_id = len(anns['image_idxs'])
        if max_img_id is not None:
            while max_id > 0 and anns['image_idxs'][max_id - 1] >= max_img_id:
                max_id -= 1

        self.obj_masks = anns['object_masks'][min_id: max_id]
        self.img_ids = anns['image_idxs'][min_id: max_id]
        self.cat_ids = anns['category_idxs'][min_id: max_id]

        if anns['feature_vectors'] is not None and anns['feature_vectors'] != []:
            self.feat_vecs = anns['feature_vectors'][min_id: max_id]
        else:
            self.feat_vecs = None

        self.img_dir = img_dir

        self.attr_names = attr_names

        self.dataset_h5 = None
        transform_list = [transforms.ToTensor()]
        self._transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.img_ids)

    def load_h5(self):
        self.dataset_h5 = h5py.File(self.img_dir, 'r')['images']

    def __getitem__(self, idx):
        if self.dataset_h5 is None:
            self.load_h5()
        img = self.dataset_h5[self.img_ids[idx]]
        img = img[:, :, ::-1].copy()  # first transform to cv2 BGR
        img = self._transform(img)

        labels = -1
        if self.feat_vecs is not None:
            labels = [self.feat_vecs[idx][attr] for attr in self.attr_names]

        mask = mask_util.decode(self.obj_masks[idx])
        bbox = np.argwhere(mask)
        mask = torch.Tensor(mask).unsqueeze(dim=0)
        seg = img * mask

        if bbox.size > 0:
            (ystart, xstart), (ystop, xstop) = bbox.min(0), bbox.max(0) + 1
            seg = seg[:, ystart:ystop, xstart:xstop]

        transform_list = [transforms.Resize((244, 224)),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        data = transforms.Compose(transform_list)(seg)
        return data, labels, idx, self.img_ids[idx]


class ObjectAttributeDataset(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        dataset = ClevrObjectDataset(self.args.ann_path, self.args.img_h5, self.args.attr_names,
                                     max_img_id=self.args.split_id)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        dataset = ClevrObjectDataset(self.args.ann_path, self.args.img_h5, self.args.attr_names,
                                     min_img_id=self.args.split_id)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        dataset = ClevrObjectDataset(self.args.ann_path, self.args.img_h5, self.args.attr_names)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return dataloader
