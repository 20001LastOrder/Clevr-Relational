from torch.utils.data import DataLoader
from .clevr_relation import RelationDataset
from .clevr_relation import ObjectRelationDataset
from .scene_based import SceneBasedRelationDataset, SceneBasedObjectRelationDataset


def get_dataset(opt):
    if opt.model_type == 'scene_based':
        return SceneBasedObjectRelationDataset(opt)
    else:
        return ObjectRelationDataset(opt)


def get_test_dataloader(opt):
    if opt.model_type == 'scene_based':
        dataset = SceneBasedRelationDataset(opt.test_ann_path, opt.test_img_h5, opt.num_rels)
    else:
        dataset = RelationDataset(opt.test_ann_path, opt.test_img_h5, opt.num_rels)

    dataloader = DataLoader(
        dataset,
        batch_size=1 if opt.model_type == 'scene_based' else opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=True
    )
    return dataloader
