import numpy as np
import pycocotools.mask as mask_util


def preprocess_rle(rle):
    """Turn ASCII string into rle bytes"""
    new_rle = {'counts': rle['counts'].encode('ASCII'), 'size': rle['size']}
    return new_rle


def rle_masks_to_boxes(masks):
    """Computes the bounding box of each mask in a list of RLE encoded masks."""
    if len(masks) == 0:
        return []

    decoded_masks = [
        np.array(mask_util.decode(rle), dtype=np.float32) for rle in masks
    ]

    def get_bounds(flat_mask):
        inds = np.where(flat_mask > 0)[0]
        return inds.min(), inds.max()

    boxes = np.zeros((len(decoded_masks), 4))
    keep = [True] * len(decoded_masks)
    for i, mask in enumerate(decoded_masks):
        if mask.sum() == 0:
            keep[i] = False
            continue
        flat_mask = mask.sum(axis=0)
        x0, x1 = get_bounds(flat_mask)
        flat_mask = mask.sum(axis=1)
        y0, y1 = get_bounds(flat_mask)
        boxes[i, :] = (x0, y0, x1, y1)

    return boxes, np.where(keep)[0]


def mask_to_bbox(mask):
    """Compute the tight bounding box of a binary mask."""
    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]
    y0 = ys[0]
    y1 = ys[-1]
    return np.array((x0, y0, x1, y1), dtype=np.float32)


def process_object_mask(anno, category_id):
    obj = {}
    rle = preprocess_rle(anno['mask'])
    bbox, _ = rle_masks_to_boxes([rle])
    obj['bbox'] = bbox[0]
    obj['segmentation'] = rle
    obj['bbox_mode'] = 0
    obj['category_id'] = category_id
    return obj


def get_category_map(categories):
    return {cat: idx for idx, cat in enumerate(categories)}


def detector_obj_to_cat():
    return lambda obj: 'object'


def detector_obj_to_cat_carla():
    return lambda obj: obj['category']