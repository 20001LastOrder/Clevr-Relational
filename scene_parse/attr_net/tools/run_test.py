import numpy as np
import torch
from scene_parse.attr_net.datasets import get_test_dataloader
from scene_parse.attr_net.model import AttrNetClassificationModule
from scene_parse.attr_net.config import AttrNetConfiguration
import h5py
from tqdm import tqdm
import json
import argparse
import yaml
import torch.nn.functional as functional


def collect_proba(preds):
    result = [[] for _ in preds[0]]  # each attribute of the preds have the same dimension (# of objects)
    for pred in preds:
        pred = functional.softmax(pred, dim=1)
        for j, value in enumerate(pred.cpu().tolist()):
            result[j].append(value)
    return result


def main(opt):
    model = AttrNetClassificationModule.load_from_checkpoint(opt.model_path, args=opt)
    dataloader = get_test_dataloader(opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(opt.attr_map_path, 'r') as f:
        attr_map = json.load(f)['attributes']

    attr_names = opt.attr_names

    num_images = len(h5py.File(opt.test_img_h5, 'r')['images'])

    scenes = [{
        'image_index': i,
        'objects': []
    } for i in range(num_images)]

    model.to(device)

    for data, _, _, img_ids in tqdm(dataloader, 'processing objects batches'):
        data = data.to(device)

        preds = model.forward(data)

        attribute_labels = np.stack([torch.argmax(pred, dim=1).cpu() for pred in preds]).transpose((1, 0)) \
            if not opt.use_proba else collect_proba(preds)

        for i, attributes in enumerate(attribute_labels):
            img_id = img_ids[i]
            obj = {}
            for j, attribute in enumerate(attributes):
                attr_name = attr_names[j]
                if not opt.use_proba:
                    obj[attr_name] = attr_map[attr_name][attribute]
                else:
                    obj[attr_name] = attribute

            scenes[img_id]['objects'].append(obj)

    with open(opt.output_path, 'w') as f:
        json.dump({'scenes': scenes}, f)
    print('Output scenes saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True)
    arguments = parser.parse_args()
    with open(arguments.config_fp) as fp:
        dataMap = yaml.safe_load(fp)

    config = AttrNetConfiguration(**dataMap)

    main(config)
