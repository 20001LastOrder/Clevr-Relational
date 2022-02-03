from typing import Dict
from tqdm import tqdm
from predictor import ObjectPredictor, get_object_predictor
import numpy as np
import pycocotools.mask as mask_util
import h5py
from models import get_pretrained_mask_rcnn
import pickle
import argparse
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import yaml

from config import ObjectDetectorTrainConfig
from datasets import set_object_recognition_dataset


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Instance features are collected
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_feats = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_feats, all_keyps


def extend_results(index, all_res, im_res, classes):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for obj_idx in range(0, len(im_res)):
        all_res[classes[obj_idx]][index].append(im_res[obj_idx])


def get_segments(pred_masks):
    segs = []
    for mask in pred_masks:
        rle = mask_util.encode(np.asfortranarray(mask, dtype=np.uint8))

        # For dumping to json, need to decode the byte string.
        # https://github.com/cocodataset/cocoapi/issues/70
        rle['counts'] = rle['counts'].decode('ascii')
        segs.append(rle)
    return segs


def append_score_to_bboxes(bboxes, scores):
    # extend the dimension of scores
    scores = scores.unsqueeze(dim=1)
    return np.hstack([bboxes, scores])


def predict_for_images(image_h5_fp, num_categories: int, predictor: ObjectPredictor) -> Dict:
    images = h5py.File(image_h5_fp, 'r')['images']
    all_boxes, all_segms, all_feats, all_keyps = empty_results(num_categories, len(images))

    for i, im in enumerate(tqdm(images)):
        # convert the image color channel to the one wrt cv2
        predictions, features = predictor(im[:, :, ::-1])
        objects = predictions['instances'].to('cpu')
        classes = objects.pred_classes.numpy()
        segments = get_segments(objects.pred_masks.numpy())
        bboxes = append_score_to_bboxes(objects.pred_boxes.tensor, objects.scores)
        extend_results(i, all_boxes, bboxes, classes)
        extend_results(i, all_segms, segments, classes)
        extend_results(i, all_feats, features.to('cpu').numpy(), classes)
    return dict(
        all_boxes=all_boxes,
        all_segms=all_segms,
        all_keyps=all_keyps,
        all_feats=all_feats,
        cfg='None'
    )


def test(cfg, model, config):
    set_object_recognition_dataset(config)
    evaluator = COCOEvaluator(config.dataset_name, output_dir='.')
    val_loader = build_detection_test_loader(cfg, config.dataset_name)

    print(inference_on_dataset(model, val_loader, evaluator))


def main(args):

    cfg = get_pretrained_mask_rcnn(args.dataset_name, args.num_categories)
    predictor = get_object_predictor(cfg, args.weight_path, args.score_threshold)

    if args.test:
        with open(args.config_fp) as f:
            data_map = yaml.safe_load(f)
        config = ObjectDetectorTrainConfig(**data_map['test'])
        test(cfg, predictor.model, config)
    else:
        results = predict_for_images(args.image_h5, args.num_categories, predictor)

        with open(args.output_fp, 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--weight_path', type=str, required=True)
    parser.add_argument('--image_h5', type=str, required=True)
    parser.add_argument('--output_fp', type=str, required=True)
    parser.add_argument('--num_categories', type=int, default=1)
    parser.add_argument('--score_threshold', type=float, default=0.5)

    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--config_fp', type=str, required=False)
    arguments = parser.parse_args()

    args = parser.parse_args()
    main(args, )
