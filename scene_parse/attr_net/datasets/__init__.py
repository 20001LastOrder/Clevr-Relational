from torch.utils.data import DataLoader
from .clevr_object import ClevrObjectDataset
from .clevr_object import ObjectAttributeDataset


def get_dataset(opt):
    if opt.dataset == 'clevr':
        ds = ObjectAttributeDataset(opt)
    else:
        raise ValueError('Invalid datsaet %s' % opt.dataset)
    return ds


def get_test_dataloader(opt):
    dataset = ClevrObjectDataset(opt.test_ann_path, opt.test_img_h5, opt.attr_names)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=True
    )
    return dataloader