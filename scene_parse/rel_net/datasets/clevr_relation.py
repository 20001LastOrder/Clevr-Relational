import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pycocotools.mask as mask_util
import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class RelationDataset(Dataset):
    def __init__(self, obj_ann_path, img_dir, num_rels, img_ids=None):
        with open(obj_ann_path) as f:
            anns = json.load(f)

        # search for the object id range corresponding to the image split
        # TODO: randomize the image selection
        img_ids = set(img_ids) if img_ids is not None else set(range(len(anns['scenes'])))

        self.scenes = anns['scenes']
        self.dataset = []
        self.img_dir = img_dir

        for rel in anns['relationships']:
            if rel['image_id'] not in img_ids:
                continue

            # add positive samples of the pair
            labels = [0 for _ in range(num_rels)]
            if 'rel_ids' in rel:
                for pos in rel['rel_ids']:
                    if pos < num_rels:
                        labels[pos] = 1

            self.dataset.append([rel['image_id'], rel['source'], rel['target'], labels])

        self.dataset_h5 = None
        self.resize_transformer = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
        transform_list = [transforms.ToTensor(), transforms.Resize((224, 224)),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self._transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.dataset)

    def load_h5(self):
        self.dataset_h5 = h5py.File(self.img_dir, 'r')['images']

    def __getitem__(self, idx):
        img_id = self.dataset[idx][0]
        source_id = self.dataset[idx][1]
        target_id = self.dataset[idx][2]
        labels = self.dataset[idx][3]

        if self.dataset_h5 is None:
            self.load_h5()

        img = self.dataset_h5[img_id]
        img = img[:, :, ::-1].copy()  # first transform to cv2 BGR
        img = self._transform(img)

        source_mask = np.expand_dims(mask_util.decode(self.scenes[img_id][source_id]['mask']).astype(int), axis=2)
        target_mask = np.expand_dims(mask_util.decode(self.scenes[img_id][target_id]['mask']).astype(int), axis=2)

        source_mask = self.resize_transformer(source_mask)
        target_mask = self.resize_transformer(target_mask)

        source = torch.cat([source_mask, img], dim=0)
        target = torch.cat([target_mask, img], dim=0)

        return source, target, np.int32(labels), source_id, target_id, img_id


class ObjectRelationDataset(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        with open(args.ann_path) as f:
            anns = json.load(f)
        idx = np.array(range(len(anns['scenes'])))
        if args.shuffle_train:
            np.random.shuffle(idx)

        self.train_idx = idx[:args.train_size]
        if self.args.val_size is None:
            self.val_idx = idx[args.train_size:]
        else:
            self.val_idx = idx[args.train_size:args.train_size + self.args.val_size]
        self.test_idx = idx[args.train_size:]

    def train_dataloader(self):
        dataset = RelationDataset(self.args.ann_path, self.args.img_h5, self.args.num_rels, self.train_idx)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        dataset = RelationDataset(self.args.ann_path, self.args.img_h5, self.args.num_rels, self.val_idx)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return dataloader

    def test_dataloader(self):
        dataset = RelationDataset(self.args.ann_path, self.args.img_h5, self.args.num_rels, self.test_idx)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return dataloader
