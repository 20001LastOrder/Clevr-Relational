from torch.utils.data import DataLoader
from .clevr_object import ClevrObjectDataset
from .clevr_object import ClevrObjectAttributeDataset

def get_dataset(opt, split):
    if opt.dataset == 'clevr':
        ds = ClevrObjectAttributeDataset(opt)
    else:
        raise ValueError('Invalid datsaet %s' % opt.dataset)
    return ds


def get_test_dataloader(opt):
    dataset = ClevrObjectDataset(opt.clevr_ann_path, opt.clevr_img_h5, opt.attr_names)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=True
    )
    return dataloader