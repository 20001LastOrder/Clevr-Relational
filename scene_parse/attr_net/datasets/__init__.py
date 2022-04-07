from torch.utils.data import DataLoader
from .clevr_object import ClevrObjectDataset, ClevrObjectDatasetCoord
from .clevr_object import ObjectAttributeDataset


def get_dataset(opt):
    return ObjectAttributeDataset(opt)


def get_test_dataloader(opt):
    dataset = ClevrObjectDatasetCoord(opt.test_ann_path, opt.test_img_h5, opt.attr_names, opt.output_dims) \
        if opt.include_coords else ClevrObjectDataset(opt.test_ann_path, opt.test_img_h5, opt.attr_names)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=True
    )
    return dataloader