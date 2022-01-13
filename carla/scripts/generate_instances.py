import cv2
import argparse
import glob
from tqdm import tqdm
import numpy as np
from typing import List, Optional
import pycocotools.mask as mask_util
import json
import re
import pandas as pd
from collections import defaultdict
from utils import get_gtl_transformer

COLOR_MAPPING = {
    'car': (23, 55)
}

OBJ_TYPE_MAPPING = {
    24: "vehicle.audi.a2",
    25: "vehicle.audi.etron",
    26: "vehicle.audi.tt",
    28: "vehicle.bmw.grandtourer",
    29: "vehicle.micro.microlino",
    31: "vehicle.chevrolet.impala",
    32: "vehicle.citroen.c3",
    34: "vehicle.dodge.charger_2020",
    38: "vehicle.jeep.wrangler_rubicon",
    39: "vehicle.lincoln.mkz_2017",
    40: "vehicle.lincoln.mkz_2020",
    41: "vehicle.mercedes.coupe",
    42: "vehicle.mercedes.coupe_2020",
    43: "vehicle.mini.cooper_s",
    44: "vehicle.mini.cooper_s_2021",
    45: "vehicle.ford.mustang",
    46: "vehicle.nissan.micra",
    50: "vehicle.seat.leon",
    52: "vehicle.tesla.model3",
    53: "vehicle.toyota.prius"
}


def find_label(encoding: int) -> Optional[str]:
    for key, (low, high) in COLOR_MAPPING.items():
        if low <= encoding <= high:
            return key
    return None


def find_ego_obj(objs):
    for obj in objs:
        if obj['ego']:
            return obj
    return None


def find_matching_object(color_id, objs):
    if color_id not in OBJ_TYPE_MAPPING:
        print(color_id)
        return None

    label = OBJ_TYPE_MAPPING[color_id]
    for obj in objs:
        if label == obj['vehicle_type']:
            return obj
    return None


def process_single_image(image, location_objs) -> List:
    """
    :param location_objs: A list storing the location information of the objects in the scene
    :param image: an image with only one channel (w, h, 1)
    :return: a list contains all included objects in the scene with the bounding box and mask
    """

    if location_objs is not None:
        ego_obj = find_ego_obj(location_objs)
        transformer = get_gtl_transformer([ego_obj['x'], ego_obj['y'], ego_obj['z']], ego_obj['rotation'])

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

        location = []
        if location_objs is not None:
            matching_obj = find_matching_object(value, location_objs)
            location = list(transformer([matching_obj['x'], matching_obj['y'], matching_obj['z']])) \
                if matching_obj is not None else []

        objects.append({
            'label': label,
            'mask':  mask,  # encoding only works with fortran array
            'location': location
        })

    return objects


def dataframe_to_dict(df, key_column):
    df = df.reset_index()
    dictionary = defaultdict(list)
    for d in df.to_dict('records'):
        dictionary[d[key_column]].append(d)
    return dictionary


def main(args):
    locations = None
    if args.location_file:
        locations = dataframe_to_dict(pd.read_csv(args.location_file), 'frame')

    image_files = glob.glob(f'{args.segmentation_folder}/{args.image_file_pattern}')
    scenes = []
    for i, filename in enumerate(tqdm(image_files)):
        # read the image and only care about the encoding channel
        image = cv2.imread(filename)[:, :, args.encoding_channel]
        image_filename = re.split('[/\\\\]', filename)[-1]
        image_id = int(image_filename.split('.')[0])
        location_dict = locations[image_id] if locations is not None else None
        objects = process_single_image(image, location_dict)
        for obj in objects:
            if obj['location'] == []:
                print(f'image_id is {image_id}')
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
    parser.add_argument("--location_file", type=str)
    parser.add_argument('--output_path', type=str, default='data/instance-segmentation/scenes.json')
    parser.add_argument('--image_file_pattern', type=str, default='*.png')
    parser.add_argument('--encoding_channel', type=int, default=2, choices=[0, 1, 2],
                        help='index of the encoding channel (0: b, 1: g, 2: r)')
    main(parser.parse_args())
