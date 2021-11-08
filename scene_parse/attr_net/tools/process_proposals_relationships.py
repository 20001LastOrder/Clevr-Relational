import os
import sys
import json
import argparse
import pickle
import pycocotools.mask as mask_util
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '.')
import utils
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='clevr', type=str)
parser.add_argument('--proposal_path', required=True, type=str)
parser.add_argument('--gt_scene_path', default=None, type=str)
parser.add_argument('--output_path', required=True, type=str)
parser.add_argument('--align_iou_thresh', default=0.7, type=float)
parser.add_argument('--score_thresh', default=0.9, type=float)
parser.add_argument('--suppression', default=0, type=int)
parser.add_argument('--suppression_iou_thresh', default=0.5, type=float)
parser.add_argument('--suppression_iomin_thresh', default=0.5, type=float)


def main(args):
    output_dir = os.path.dirname(args.output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    scenes = None
    if args.gt_scene_path is not None:
        with open(args.gt_scene_path) as f:
            scenes = json.load(f)['scenes']
    with open(args.proposal_path, 'rb') as f:
        proposals = pickle.load(f)
    segms = proposals['all_segms']
    boxes = proposals['all_boxes']
    features = proposals['all_feats']

    nimgs = len(segms[0])
    ncats = len(segms)
    img_anns = []

    for i in tqdm(range(nimgs), 'processing images...'):
        obj_anns = []
        for c in range(ncats):
            for j, m in enumerate(segms[c][i]):
                if boxes[c][i][j][4] > args.score_thresh:
                    mask = mask_util.decode(m)
                    for k, o in enumerate(scenes[i]['objects']):
                        mask_gt = mask_util.decode(o['mask'])
                        if utils.iou(mask, mask_gt) > args.align_iou_thresh:
                            obj_ann = {
                                'mask': m,
                                'image_idx': i,
                                'obj_idx': k,
                                'score': float(boxes[c][i][j][4]),
                            }
                            obj_anns.append(obj_ann)
                            break
        img_anns.append(obj_anns)

    if scenes is None and args.suppression:
        # Apply suppression on test proposals
        all_objs = []
        for i, img_ann in enumerate(tqdm(img_anns, 'suppression images...')):
            objs_sorted = sorted(img_ann, key=lambda k: k['score'], reverse=True)
            objs_suppressed = []
            for obj_ann in objs_sorted:
                if obj_ann['score'] > args.score_thresh:
                    duplicate = False
                    for obj_exist in objs_suppressed:
                        mo = mask_util.decode(obj_ann['mask'])
                        me = mask_util.decode(obj_exist['mask'])
                        if utils.iou(mo, me) > args.suppression_iou_thresh \
                           or utils.iomin(mo, me) > args.suppression_iomin_thresh:
                            duplicate = True
                            break
                    if not duplicate:
                        objs_suppressed.append(obj_ann)
            all_objs += objs_suppressed
            print('| running suppression %d / %d images' % (i+1, nimgs))
    else:
        all_objs = [obj_ann for img_ann in img_anns for obj_ann in img_ann]

    scene_objects = [[None] * len(scene['objects']) for scene in scenes]
    
    for obj in all_objs:
        scene_objects[obj['image_idx']][obj['obj_idx']] = obj
    
    dir_label_map = {
        'front': 0,
        'behind': 1,
        'left': 0,
        'right': 1
    }
    
    horizontal_relationships = []
    vertical_relationships = []
    
    if scenes is not None:
        for i, scene in enumerate(scenes):
            for rel in scene['relationships']:
                for source, targets in enumerate(scene['relationships'][rel]):
                    # add the relationship only if the detector has recongnized the object
                    if scene_objects[i][source] == None:
                        continue
                    for target in targets:
                        if scene_objects[i][target] == None:
                            continue
                        # append the relationship
                        if rel in ['left', 'right']:
                            horizontal_relationships.append({
                                'image_id': i,
                                'source': source,
                                'target': target,
                                'label': dir_label_map[rel] 
                            })
                        else:
                            vertical_relationships.append({
                                'image_id': i,
                                'source': source,
                                'target': target,
                                'label': dir_label_map[rel] 
                            })
    
    output = {
        'scenes': scene_objects,
        'horizontal_labels': horizontal_relationships,
        'vertical_labels': vertical_relationships
    }
    
    print('| saving object annotations to %s' % args.output_path)
    with open(args.output_path, 'w') as fout:
        json.dump(output, fout)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)