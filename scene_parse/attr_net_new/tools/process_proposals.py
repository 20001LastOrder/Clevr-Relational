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
parser.add_argument('--output_path_rel', required=True, type=str)
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
                    if scenes is None: # no ground truth alignment
                        obj_ann = {
                            'mask': m,
                            'image_idx': i,
                            'category_idx': c,
                            'feature_vector': None,
                            'score': float(boxes[c][i][j][4]),
                            'features': features[c][i][j].tolist()
                        }
                        obj_anns.append(obj_ann)
                    else:
                        mask = mask_util.decode(m)
                        for k, o in enumerate(scenes[i]['objects']):
                            mask_gt = mask_util.decode(o['mask'])
                            if utils.iou(mask, mask_gt) > args.align_iou_thresh:
                                if args.dataset == 'clevr':
                                    vec = utils.get_feat_vec_clevr(o)
                                else:
                                    vec = utils.get_feat_vec_mc(o)
                                obj_ann = {
                                    'mask': m,
                                    'image_idx': i,
                                    'category_idx': c,
                                    'obj_idx': k,
                                    'feature_vector': vec,
                                    'score': float(boxes[c][i][j][4]),
                                    'features': features[c][i][j].tolist()
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
    else:
        all_objs = [obj_ann for img_ann in img_anns for obj_ann in img_ann]

    obj_masks = [o['mask'] for o in all_objs]
    img_ids = [o['image_idx'] for o in all_objs]
    cat_ids = [o['category_idx'] for o in all_objs]
    scores = [o['score'] for o in all_objs]
    features = [o['features'] for o in all_objs]
    
    if scenes is not None:
        feat_vecs = [o['feature_vector'] for o in all_objs]
    else:
        feat_vecs = []
    output = {
        'object_masks': obj_masks,
        'image_idxs': img_ids,
        'category_idxs': cat_ids,
        'feature_vectors': feat_vecs,
        'scores': scores,
        'features': features
    }
    print('| saving object annotations to %s' % args.output_path)
    with open(args.output_path, 'w') as fout:
        json.dump(output, fout)
    
    scene_objects = [[None] * len(scene['objects']) for scene in scenes] if scenes != None else [[] for _ in range(nimgs)]
        
    for obj in all_objs:
        if scenes is not None:
            scene_objects[obj['image_idx']][obj['obj_idx']] = obj
        else:
            scene_objects[obj['image_idx']].append(obj)
    
    dir_label_map = {
        'front': 0,
        'behind': 1,
        'left': 0,
        'right': 1
    }

    relationships = []

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
    else:
        for i, objects in enumerate(scene_objects):
            num_obj = len(objects)
            for source in range(num_obj - 1):
                for target in range(source + 1, num_obj):
                    relationships.append({
                            'image_id': i,
                            'source': source,
                            'target': target
                    })
                    relationships.append({
                            'image_id': i,
                            'source': target,
                            'target': source
                    })
            

    output = {
        'scenes': scene_objects,
        'horizontal_labels': relationships,
        'vertical_labels': relationships
    }

    print('| saving object annotations to %s' % args.output_path_rel)
    with open(args.output_path_rel, 'w') as fout:
        json.dump(output, fout)
            
        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)