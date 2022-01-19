import argparse
import glob
from tqdm import tqdm
import pycocotools.mask as mask_util
from utils import get_gtl_transformer, left_rel, right_rel, before_rel, behind_rel, read_json, write_json
import numpy as np
import cv2


REL_MAP = {
    'left': left_rel,
    'right': right_rel,
    'before': before_rel,
    'behind': behind_rel
}

TYPE_IDS = {
    10: 'vehicle',
    4: 'pedestrian',
    18: 'traffic_light'
}


def get_obj_with_type(objects, type_id):
    return [obj for obj in objects if len(obj['semantic_tags']) and obj['semantic_tags'][0] == type_id]


def get_visible_objects(image, objects, type_id, min_pixel):
    type_objects = get_obj_with_type(objects, type_id)
    result = []
    for obj in type_objects:
        color = np.array([type_id, obj['green'], obj['blue']])
        segmentation = (image == color).all(axis=2)

        if segmentation.sum() < min_pixel:
            continue
        mask = mask_util.encode(np.asfortranarray(segmentation))
        mask['counts'] = mask['counts'].decode('ASCII')
        obj['mask'] = mask

        obj['category'] = TYPE_IDS[type_id]
        result.append(obj)
    return result


def get_camera(objects):
    for obj in objects:
        if obj['type'] == 'camera':
            return obj


def to_local_coord(objects, transformer):
    for obj in objects:
        location = obj['location']
        obj['location'] = list(transformer([location['x'], location['y'], location['z']]))
    return objects


def process_relationships(objects):
    n = len(objects)
    relationships = {rel: [[] for _ in range(n)] for rel in REL_MAP.keys()}

    for i in range(n):
        for j in range(i + 1, n):
            for rel, check in REL_MAP.items():
                if check(objects[i]['location'], objects[j]['location']):
                    relationships[rel][i].append(j)
                if check(objects[j]['location'], objects[i]['location']):
                    relationships[rel][j].append(i)

    return relationships


def main(args):
    image_names = glob.glob(f'{args.img_dir}/*.png')
    scenes = []
    for image_path in tqdm(image_names):
        image_name = image_path.split('\\')[-1].split('.')[0]
        image = cv2.imread(image_path)[:, :, ::-1]
        objects = read_json(f'{args.scenes_folder}/{int(image_name)}.json')
        visible_objects = []
        for type_id in TYPE_IDS:
            visible_objects.extend(get_visible_objects(image, objects, type_id, args.min_pixel))

        camera = get_camera(objects)
        transformer = get_gtl_transformer((camera['location']['x'], camera['location']['y'], camera['location']['z']), camera['rotation']['yaw'])
        to_local_coord(visible_objects, transformer)

        relationships = process_relationships(visible_objects)

        scenes.append({
            'objects': visible_objects,
            'image_index': int(image_name),
            'image_filename': f'{image_name}.png',
            'relationships': relationships
        })

    write_json({'scenes': scenes}, args.output_fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ../data/carla/instance_segmentation/*.png
    parser.add_argument('--img_dir', required=True, type=str)
    # ../data/carla/scene_graph/
    parser.add_argument('--scenes_folder', type=str, required=True)
    parser.add_argument('--output_fp', type=str, required=True)
    parser.add_argument('--min_pixel', type=int, default=50, help='minimum number of pixels for an object to be '
                                                                  'considered as visible')

    main(parser.parse_args())
