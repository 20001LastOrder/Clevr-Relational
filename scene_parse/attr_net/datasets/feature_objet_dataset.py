import pickle

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import torch


class FeatureObjectDataset(Dataset):
    def __init__(self, obj_ann_path, attr_names, min_img_id=None, max_img_id=None):
        with open(obj_ann_path, 'rb') as f:
            dictionary = pickle.load(f)
            anns = dictionary['objects']
            self.image_features = dictionary['img_features']
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
        self.features = anns['features'][min_id: max_id]

        if anns['feature_vectors'] is not None and anns['feature_vectors'] != []:
            self.feat_vecs = anns['feature_vectors'][min_id: max_id]
        else:
            self.feat_vecs = None

        self.attr_names = attr_names

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_feature = self.image_features[self.img_ids[idx]].squeeze()
        obj_feature = torch.tensor(self.features[idx])
        labels = -1
        if self.feat_vecs is not None:
            labels = [self.feat_vecs[idx][attr] for attr in self.attr_names]

        return img_feature, obj_feature, labels, idx, self.img_ids[idx]


class FeatureObjectDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        dataset = FeatureObjectDataset(self.args.ann_path, self.args.attr_names,
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
        dataset = FeatureObjectDataset(self.args.ann_path, self.args.attr_names,
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
        dataset = FeatureObjectDataset(self.args.ann_path, self.args.attr_names)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return dataloader
