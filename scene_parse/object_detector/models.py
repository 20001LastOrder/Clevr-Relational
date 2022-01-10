from detectron2.config import get_cfg
from detectron2 import model_zoo


def get_pretrained_mask_rcnn(dataset_train: str, num_classes: int, max_iter: int=200):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.INPUT.MASK_FORMAT = 'bitmask'

    return cfg
