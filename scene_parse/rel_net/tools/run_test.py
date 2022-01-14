import argparse

import yaml

from scene_parse.rel_net.config import RelNetConfiguration
from scene_parse.rel_net.datasets import get_test_dataloader
from scene_parse.rel_net.model import RelNetModule
import torch
from tqdm import tqdm
import json


def main(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RelNetModule.load_from_checkpoint(opt.model_path, args=opt)
    dataloader = get_test_dataloader(opt)

    with open(opt.scenes_path, 'r') as f:
        result = json.load(f)

    scenes = result['scenes']
    relation_names = opt.label_names

    for scene in scenes:
        if 'relationships' not in scene:
            scene['relationships'] = {}
        for relation in relation_names:
            scene['relationships'][relation] = [[] for _ in scene['objects']]

    model.to(device)

    for sources, targets, _, source_ids, target_ids, img_ids in tqdm(dataloader, 'processing objects batches'):
        sources = sources.to('cuda')
        targets = targets.to('cuda')
        preds = model.forward(sources, targets)

        if not opt.use_proba:
            preds = torch.round(preds).int().cpu().numpy()

        for i, label in enumerate(preds):
            img_id = img_ids[i]
            edge = (source_ids[i].item(), target_ids[i].item())
            for j, pred in enumerate(preds[i]):
                relation = relation_names[j]
                if not opt.use_proba and pred[i][j].item() == 1:
                    scenes[img_id]['relationships'][relation][edge[0]].append(edge[1])
                else:
                    scenes[img_id]['relationships'][relation][edge[0]].append((edge[1], preds[i][j].item()))

    with open(opt.output_path, 'w') as f:
        json.dump({'scenes': scenes}, f)
    print('Output scenes saved!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True)
    arguments = parser.parse_args()

    with open(arguments.config_fp) as fp:
        dataMap = yaml.safe_load(fp)

    config = RelNetConfiguration(**dataMap)
    main(config)
