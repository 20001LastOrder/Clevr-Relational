from torch.utils.data import DataLoader
from .clevr_relation import ClevrRelationDataset
from .clevr_relation import ObjectRelationDataset


def get_dataset(opt):
    return ObjectRelationDataset(opt)


def get_test_dataloader(opt):
    dataset = ClevrRelationDataset(opt.clevr_ann_path, opt.clevr_img_h5, opt.label_name)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=True
    )
    return dataloader
