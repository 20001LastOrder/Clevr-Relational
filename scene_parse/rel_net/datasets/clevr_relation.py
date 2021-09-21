import os
import json

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pycocotools.mask as mask_util
import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class ClevrRelationDataset(Dataset):

    def __init__(self, obj_ann_path, img_dir, label,
                 min_img_id=None, max_img_id=None):
        with open(obj_ann_path) as f:
            anns = json.load(f)

        # search for the object id range corresponding to the image split
        min_id = 0
        if min_img_id is not None:
            while anns[label][min_id]['image_id'] < min_img_id:
                min_id += 1
        max_id = len(anns[label])
        if max_img_id is not None:
            while max_id > 0 and anns[label][max_id - 1]['image_id'] >= max_img_id:
                max_id -= 1
        
        self.scenes = anns['scenes']
        self.dataset = anns[label][min_id: max_id]

        self.img_dir = img_dir
        
        self.dataset_h5 = None
        transform_list = [transforms.ToTensor()]
        self._transform = transforms.Compose(transform_list)
        
    def __len__(self):
        return len(self.dataset)

    def load_h5(self):
        self.dataset_h5 = h5py.File(self.img_dir, 'r')['images']

    def __getitem__(self, idx):
        img_id = self.dataset[idx]['image_id']
        label = self.dataset[idx]['label'] if 'label' in self.dataset[idx] else -1
        source = self.dataset[idx]['source']
        target = self.dataset[idx]['target']
        
        if self.dataset_h5 is None:
            self.load_h5()
        img = self.dataset_h5[img_id]
        img = img[:, :, ::-1].copy() # first transform to cv2 BGR
        img = self._transform(img)
        
        source_mask = mask_util.decode(self.scenes[img_id][source]['mask']).astype(int)
        target_mask = mask_util.decode(self.scenes[img_id][target]['mask']).astype(int)
        
        combined_mask = source_mask + target_mask
        bbox = np.argwhere(combined_mask)
        (ystart, xstart), (ystop, xstop) = bbox.min(0), bbox.max(0) + 1 
        
        mask = source_mask - target_mask
        mask = torch.Tensor(mask).unsqueeze(dim=0)
        
        transform_list = [transforms.Resize((244, 224))]
                          #transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]

        img = transforms.Compose(transform_list)(mask)
        return img, label, source, target, img_id

class ClevrObjectRelationDataset(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        dataset = ClevrRelationDataset(self.args.clevr_ann_path, self.args.clevr_img_h5, self.args.label_name,
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
        dataset = ClevrRelationDataset(self.args.clevr_ann_path, self.args.clevr_img_h5, self.args.label_name,
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
