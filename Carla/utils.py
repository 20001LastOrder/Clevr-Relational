from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.projects import point_rend
import torch
import json
import matplotlib.pyplot as plt


def get_panoptic_pretrained_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return cfg, predictor


def get_semantic_pretrained_predictor():
    cfg = get_cfg()
    # Add PointRend-specific config
    point_rend.add_pointrend_config(cfg)
    # Load a config from file
    cfg.merge_from_file(
        "C:\\Users\\chenp\\Documents\\github\\detectron2\\projects\\PointRend\\configs\\SemanticSegmentation\\pointrend_semantic_R_101_FPN_1x_cityscapes.yaml")

    cfg.MODEL.WEIGHTS = "detectron2://PointRend/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes/202576688/model_final_cf6ac1.pkl"
    predictor = DefaultPredictor(cfg)

    return cfg, predictor


def get_instance_pretrained_predictor():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return cfg, predictor


def panoptic_segmentation(im, predictor, cfg):
    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    return out


def semantic_segmentation(im, predictor, cfg):
    outputs = predictor(im)
    pred = torch.argmax(outputs["sem_seg"].cpu(), dim=0)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_sem_seg(pred)
    return out


def predict_instance_segmentation(im, predictor, cfg):
    outputs = predictor(im)
    print(outputs['instances'])
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out


def read_json(fname):
    with open(fname) as f:
        return json.load(f)


def cv2_imshow(im):
    plt.figure(figsize=(16, 12), dpi=80)
    plt.imshow(im[:, :, ::-1])
