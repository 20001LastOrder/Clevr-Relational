import os
import argparse
import pycocotools.mask as mask_util
from scene_parse.attr_net import utils
from tqdm import tqdm


def find_gt_object(m, gt_masks, align_iou_thresh):
    """
    Given a mask and a list of ground truth mask, find closest ground truth mask higher than the threshold
    """
    mask = mask_util.decode(m)
    candidates = [(k, utils.iou(mask, mask_gt)) for k, mask_gt in enumerate(gt_masks)]
    # sort the candidates descending by the overlap between mask and the ground truth mask
    candidates = sorted(candidates, key=lambda cell: cell[1], reverse=True)
    # only return the candidate if the candidate has the overlap with the ground truth mask higher than the threshold
    return candidates[0][0] if candidates[0][1] > align_iou_thresh else -1


def main(args):
    output_dir = os.path.dirname(args.output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    attribute_map = utils.read_json(args.attribute_map)

    scenes = None
    if args.gt_scene_path is not None:
        scenes = utils.read_json(args.gt_scene_path)['scenes']

    proposals = utils.load_pickle(args.proposal_path)
    segms = proposals['all_segms']
    boxes = proposals['all_boxes']
    features = proposals['all_feats']

    n_imgs = len(segms[0])
    n_cats = len(segms)
    img_anns = []

    for i in tqdm(range(n_imgs), 'processing images...'):
        obj_anns = []
        gt_masks = [mask_util.decode(o['mask']) for o in scenes[i]['objects']] if scenes is not None else []
        for c in range(n_cats):
            for j, m in enumerate(segms[c][i]):
                score = boxes[c][i][j][4]
                # only consider objects having a detection confidence higher than the threshold
                if score < args.score_thresh:
                    continue

                obj_ann = {
                    'mask': m,
                    'image_idx': i,
                    'category_idx': c,
                    'score': float(boxes[c][i][j][4]),
                    'features': features[c][i][j].tolist()
                }
                if scenes is None:  # no ground truth alignment
                    obj_ann['feature_vector'] = None
                    obj_anns.append(obj_ann)
                else:
                    gt_object_idx = find_gt_object(m, gt_masks, args.align_iou_thresh)
                    # add the information about the ground truth object if we successfully find one
                    if gt_object_idx >= 0:
                        vec = utils.get_feat_vec(scenes[i]['objects'][gt_object_idx], attribute_map['attributes'])
                        obj_ann['feature_vector'] = vec
                        obj_ann['obj_idx'] = gt_object_idx
                        obj_anns.append(obj_ann)
        img_anns.append(obj_anns)

    if scenes is None and args.suppression:
        # Apply suppression on test proposals
        all_objs = []
        for i, img_ann in enumerate(tqdm(img_anns, 'suppression images...')):
            objs_sorted = sorted(img_ann, key=lambda k: k['score'], reverse=True)
            objs_suppressed = []
            for obj_ann in objs_sorted:
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
    feat_vecs = [o['feature_vector'] for o in all_objs] if scenes is not None else []

    output_objects = {
        'object_masks': obj_masks,
        'image_idxs': img_ids,
        'category_idxs': cat_ids,
        'feature_vectors': feat_vecs,
        'scores': scores,
        'features': features
    }

    scene_objects = [[None] * len(scene['objects']) for scene in scenes] if scenes is not None else [[] for _ in
                                                                                                     range(n_imgs)]
    for obj in all_objs:
        if scenes is not None:
            scene_objects[obj['image_idx']][obj['obj_idx']] = obj
        else:
            scene_objects[obj['image_idx']].append(obj)

    dir_label_map = {
        label: (i, rel_type) for rel_type, values in attribute_map['relationships'].items() for i, label in
        enumerate(values)
    }

    relationships = {
        key: [] for key in attribute_map['relationships']
    }

    # process relationships
    if scenes is not None:
        for i, scene in enumerate(scenes):
            for rel in scene['relationships']:
                for source, targets in enumerate(scene['relationships'][rel]):
                    # add the relationship only if the detector has recongnized the object
                    if scene_objects[i][source] is None:
                        continue
                    for target in targets:
                        if scene_objects[i][target] is None:
                            continue
                        # append the relationship
                        idx, rel_type = dir_label_map[rel]
                        relationships[rel_type].append({
                                'image_id': i,
                                'source': source,
                                'target': target,
                                'label': idx
                            })
    else:
        for i, objects in enumerate(scene_objects):
            num_obj = len(objects)
            for source in range(num_obj - 1):
                for target in range(source + 1, num_obj):
                    for rels in relationships.values():
                        rels.append({
                            'image_id': i,
                            'source': source,
                            'target': target
                        })
                        rels.append({
                            'image_id': i,
                            'source': target,
                            'target': source
                        })

    output = {
        'scenes': scene_objects,
        'objects': output_objects
    }

    for key, rels in relationships.items():
        output[key] = rels

    print('| saving object annotations to %s' % args.output_path)
    utils.write_json(output, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute_map', default='data/clevr_attribute_map.json', type=str)
    parser.add_argument('--proposal_path', required=True, type=str)
    parser.add_argument('--gt_scene_path', default=None, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--align_iou_thresh', default=0.7, type=float)
    parser.add_argument('--score_thresh', default=0.9, type=float)
    parser.add_argument('--suppression', default=0, type=int)
    parser.add_argument('--suppression_iou_thresh', default=0.5, type=float)
    parser.add_argument('--suppression_iomin_thresh', default=0.5, type=float)

    main(parser.parse_args())
