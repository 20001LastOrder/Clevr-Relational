import cv2
import argparse
import glob
from tqdm import tqdm
import numpy as np
from typing import List, Optional
import pycocotools.mask as mask_util
import json
import re


COLOR_MAPPING = {
    'car': (23, 55)
}


def find_label(encoding: int) -> Optional[str]:
    for key, (low, high) in COLOR_MAPPING.items():
        if low <= encoding <= high:
            return key
    return None


def process_single_image(image) -> List:
    """
    :param image: an image with only one channel (w, h, 1)
    :return: a list contains all included objects in the scene with the bounding box and mask
    """

    # find all unique value encoding with assigned labels
    unique_values = np.unique(image)
    objects = []
    for value in unique_values:
        label = find_label(value)
        if label is None:
            continue
        mask = (image == value)
        mask = mask_util.encode(np.asfortranarray(mask))
        mask['counts'] = mask['counts'].decode('ASCII')

        objects.append({
            'label': label,
            'mask':  mask  # encoding only works with fortran array
        })

    return objects


def main(args):
    image_files = glob.glob(f'{args.segmentation_folder}/{args.image_file_pattern}')
    scenes = []
    for i, filename in enumerate(tqdm(image_files)):
        # read the image and only care about the encoding channel
        image = cv2.imread(filename)[:, :, args.encoding_channel]
        objects = process_single_image(image)
        scenes.append({
            'objects': objects,
            'image_filename': re.split('[/\\\\]', filename)[-1],  # the filename is absolute, only need the last part
            'image_index': i
        })

    with open(args.output_path, 'w') as f:
        json.dump({'scenes': scenes}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation_folder', type=str, default='data/instance-segmentation/semantic')
    parser.add_argument('--output_path', type=str, default='data/instance-segmentation/scenes.json')
    parser.add_argument('--image_file_pattern', type=str, default='*.png')
    parser.add_argument('--encoding_channel', type=int, default=2, choices=[0, 1, 2],
                        help='index of the encoding channel (0: b, 1: g, 2: r)')
    main(parser.parse_args())
