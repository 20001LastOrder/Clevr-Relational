from scene_parse.object_detector.models import get_pretrained_mask_rcnn
from scene_parse.object_detector.datasets import ObjectDetectorDataset
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
import argparse
import yaml
from config import ObjectDetectorTrainConfig
from datasets import set_object_recognition_dataset


def config_train(cfg, output_path, dataset_train, dataset_test, max_iter=200, num_workers=4, ims_per_batch=2,
                 base_lr=0.001, batch_size_per_image=50):
    # TODO: achieve the same configuration with detectron2 configuration file
    # set training configuration
    cfg.OUTPUT_DIR = output_path
    cfg.DATASETS.TRAIN = (dataset_train,)
    cfg.DATASETS.TEST = () if dataset_test is None else (dataset_test,)
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.BASE_LR_END = 0.0
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []  # do not decay learning rate yet
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


def train_object_detector(detector_config: ObjectDetectorTrainConfig):
    categories = set_object_recognition_dataset(detector_config)

    cfg = get_pretrained_mask_rcnn(detector_config.dataset_name, len(categories))
    config_train(cfg, detector_config.output_dir, detector_config.dataset_name, None, detector_config.max_iter,
                 detector_config.num_workers, detector_config.ims_per_batch, detector_config.base_lr,
                 detector_config.batch_size_per_image)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=detector_config.resume)
    trainer.train()
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True)
    arguments = parser.parse_args()

    with open(arguments.config_fp) as f:
        dataMap = yaml.safe_load(f)

    config = ObjectDetectorTrainConfig(**dataMap['train'])

    # train the object detector
    train_object_detector(config)
