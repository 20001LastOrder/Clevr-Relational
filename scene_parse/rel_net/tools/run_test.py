import argparse

import yaml

from scene_parse.rel_net.config import RelNetConfiguration
from scene_parse.rel_net.constraints import build_adjacency_matrix, adj_probability, \
    build_adjacency_matrix_from_edge_list
from scene_parse.rel_net.datasets import get_test_dataloader
from scene_parse.rel_net.models import RelNetModule, SceneBasedRelNetModule
import torch
from tqdm import tqdm
import json


def predict_pair_based(opt, model, dataloader, scenes, relation_names, device):
    for sources, targets, _, source_ids, target_ids, img_ids in tqdm(dataloader, 'processing objects batches'):
        sources = sources.to(device)
        targets = targets.to(device)
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


def predict_scene_based(opt, model, dataloader, scenes, relation_names, device):
    for data, sources, targets, labels, image_id, (num_nodes, num_edges), _ in tqdm(dataloader, 'processing objects batches'):
        num_nodes = num_nodes.item()
        num_edges = num_edges.item()

        data = data[:num_nodes].squeeze(dim=0).to(device)
        sources = sources[:, :num_edges].squeeze(dim=0).to(device).long()
        targets = targets[:, :num_edges].squeeze(dim=0).to(device).long()

        img_id = image_id.item()

        preds = model.forward(data, sources, targets)

        if not opt.use_proba:
            preds = torch.round(preds).int().cpu().numpy()

        for i, label in enumerate(preds):
            edge = (sources[i].item(), targets[i].item())
            for j, pred in enumerate(preds[i]):
                relation = relation_names[j]
                if not opt.use_proba:
                    if preds[i][j].item() == 1:
                        scenes[img_id]['relationships'][relation][edge[0]].append(edge[1])
                else:
                    scenes[img_id]['relationships'][relation][edge[0]].append((edge[1], preds[i][j].item()))


def predict_scene_adj_based(opt, model, dataloader, scenes, relation_names, relation_map):
    for data, sources, targets, labels, image_id, (num_nodes, num_edges), _ in tqdm(dataloader,
                                                                                    'processing objects batches'):
        num_nodes = num_nodes.item()
        num_edges = num_edges.item()

        data = data[:num_nodes].squeeze(dim=0).to('cuda')
        sources = sources[:, :num_edges].squeeze(dim=0).to('cuda').long()
        targets = targets[:, :num_edges].squeeze(dim=0).to('cuda').long()

        img_id = image_id.item()

        preds = model.forward(data, sources, targets)

        adj = build_adjacency_matrix_from_edge_list(sources, targets, preds, num_nodes)
        adj = adj_probability(adj)

        if not opt.use_proba:
            adj = torch.round(preds).int().cpu().numpy()

        for i, _ in enumerate(preds):
            edge = (sources[i].item(), targets[i].item())
            prediction = adj[:, edge[0], [edge[1]]]
            for j, pred in enumerate(prediction):
                relation = relation_names[j]
                opposite_relation = relation_map[relation]
                pred = pred.item()
                if not opt.use_proba and pred == 1:
                    scenes[img_id]['relationships'][relation][edge[0]].append(edge[1])
                    scenes[img_id]['relationships'][opposite_relation][edge[1]].append(edge[0])
                else:
                    scenes[img_id]['relationships'][relation][edge[0]].append((edge[1], pred))
                    scenes[img_id]['relationships'][opposite_relation][edge[0]].append((edge[1], 1 - pred))

def main(opt):
    relation_map = None # {'left': 'right', 'front': 'behind'}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SceneBasedRelNetModule.load_from_checkpoint(opt.model_path, args=opt) if opt.model_type == 'scene_based' \
        else RelNetModule.load_from_checkpoint(opt.model_path, args=opt)

    dataloader = get_test_dataloader(opt)

    with open(opt.scenes_path, 'r') as f:
        result = json.load(f)

    scenes = result['scenes']
    relation_names = opt.label_names

    for scene in scenes:
        if 'relationships' not in scene:
            scene['relationships'] = {}

        if relation_map is None:
            for relation in relation_names:
                scene['relationships'][relation] = [[] for _ in scene['objects']]
        else:
            for relation in relation_map.keys() + relation_map.values():
                scene['relationships'][relation] = [[] for _ in scene['objects']]

    model.to(device)
    model = model.eval()

    if opt.model_type == 'scene_based':
        if relation_map is None:
            predict_scene_based(opt, model, dataloader, scenes, relation_names, device)
        else:
            predict_scene_adj_based(opt, model, dataloader, scenes, relation_names, relation_map, device)
    else:
        predict_pair_based(opt, model, dataloader, scenes, relation_names, device)

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
