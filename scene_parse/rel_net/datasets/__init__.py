from torch.utils.data import DataLoader
from .clevr_relation import ClevrRelationDataset, RelationDataset
from .clevr_relation import ObjectRelationDataset


def get_dataset(opt):
    return ObjectRelationDataset(opt)


def get_test_dataloader(opt):
    dataset = RelationDataset(opt.test_ann_path, opt.test_img_h5, opt.num_rels)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=True
    )
    return dataloader
