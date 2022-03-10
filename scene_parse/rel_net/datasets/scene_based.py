import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pycocotools.mask as mask_util
import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from collections import defaultdict


class SceneBasedRelationDataset(Dataset):
    def __init__(self, obj_ann_path, img_dir, num_rels, img_ids=None, used_rels=None):
        """

        :param obj_ann_path:
        :param img_dir:
        :param num_rels: The number of relationships to consider, this will be ignored if used_rels is None
        :param img_ids:
        :param used_rels:
        """
        with open(obj_ann_path) as f:
            anns = json.load(f)

        # search for the object id range corresponding to the image split
        img_ids = set(img_ids) if img_ids is not None else set(range(len(anns['scenes'])))

        self.scenes = anns['scenes']
        self.dataset = defaultdict(list)
        self.img_dir = img_dir
        self.num_rels = num_rels if used_rels is None else len(used_rels)

        # cannot use scenes directly since the object detector may not detected all objects
        for rel in anns['relationships']:
            if rel['image_id'] not in img_ids:
                continue

            # add positive samples of the pair
            labels = [0 for _ in range(self.num_rels)]
            if 'rel_ids' in rel:
                for pos in rel['rel_ids']:
                    if used_rels is None:
                        labels[pos] = 1
                    elif pos in used_rels:
                        labels[used_rels.index(pos)] = 1

            self.dataset[rel['image_id']].append([rel['source'], rel['target'], labels])

        self.dataset = list(self.dataset.items())
        # for image_id, edges in self.dataset:
        #     self.random_flip(edges)

        self.dataset_h5 = None
        self.resize_transformer = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
        transform_list = [transforms.ToTensor(), transforms.Resize((224, 224)),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self._transform = transforms.Compose(transform_list)

        # assume only 10% of data is labeled
        index = [i for i, _ in self.dataset]
        np.random.shuffle(index)
        self.labelled = set(index[: int(len(index) * 0.1)])

    def random_flip(self, edges):
        n = len(edges)
        indices = list(range(n))
        np.random.shuffle(indices)

        selected = indices[:int(n * self.noise_ratio)]

        # flip the relationship of this pair
        for index in selected:
            edges[index][2] = [1 - label for label in edges[index][2]]

    def __len__(self):
        return len(self.dataset)

    def load_h5(self):
        self.dataset_h5 = h5py.File(self.img_dir, 'r')['images']

    def __getitem__(self, idx):
        if self.dataset_h5 is None:
            self.load_h5()

        img_id = self.dataset[idx][0]
        edges = self.dataset[idx][1]
        max_obj = 12

        sources = torch.empty((max_obj ** 2))
        targets = torch.empty((max_obj ** 2))
        labels = torch.empty((max_obj ** 2, self.num_rels))

        num_edges = len(edges)
        sources[:num_edges] = torch.tensor([rel[0] for rel in edges])
        targets[:num_edges] = torch.tensor([rel[1] for rel in edges])
        labels[:num_edges] = torch.tensor([rel[2] for rel in edges])

        img = self._transform(self.dataset_h5[img_id])
        num_obj = len(self.scenes[img_id])

        data = torch.empty((max_obj, 4, 224, 224))
        for i, obj in enumerate(self.scenes[img_id]):
            if obj is not None:
                mask = self.resize_transformer(np.expand_dims(mask_util.decode(obj['mask']).astype(int), axis=2))
            else:
                mask = torch.zeros((1, img.shape[1], img.shape[2]))
            data[i] = torch.cat([img, mask], dim=0)

        return data, sources, targets, labels, img_id, (num_obj, num_edges), True  #if img_id in self.labelled else False


class SceneBasedObjectRelationDataset(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        with open(args.ann_path) as f:
            anns = json.load(f)
        idx = np.array(range(len(anns['scenes'])))
        np.random.shuffle(idx)

        self.train_idx = idx[:args.train_size]
        self.test_idx = idx[args.train_size:]

    def train_dataloader(self):
        dataset = SceneBasedRelationDataset(self.args.ann_path, self.args.img_h5, self.args.num_rels, self.train_idx,
                                            used_rels=self.args.used_rels)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        dataset = SceneBasedRelationDataset(self.args.ann_path, self.args.img_h5, self.args.num_rels, self.test_idx,
                                            used_rels=self.args.used_rels)
        dataloader = DataLoader(
            dataset,
            batch_size= 1 if self.args.model_type == 'scene_based' else self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
