from detectron2.config import get_cfg
from detectron2 import model_zoo


def get_pretrained_mask_rcnn(dataset_train, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (dataset_train,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 8000
    cfg.SOLVER.STEPS = []  # do not decay learning rate yet
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 50
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.INPUT.MASK_FORMAT = 'bitmask'

    return cfg
