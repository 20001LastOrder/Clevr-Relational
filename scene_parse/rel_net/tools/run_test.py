import sys
import torch
import numpy as np
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '.')
from options import get_options
from datasets import get_test_dataloader
from model import get_model, RelNetClassificationModule
from pytorch_lightning import Trainer
from torch import nn
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint
import h5py
from tqdm import tqdm
import json

opt = get_options('test')
model = RelNetClassificationModule.load_from_checkpoint(opt.model_path)
dataloader = get_test_dataloader(opt)

with open(opt.attr_map_path, 'r') as f:
    attr_map = json.load(f)

with open(opt.scenes_path, 'r') as f:
    result = json.load(f)

scenes = result['scenes']
relation_names = attr_map[opt.label_name]

for scene in scenes:
    if 'relationships' not in scene:
        scene['relationships'] = {}
    for relation in relation_names:
        scene['relationships'][relation] = [[] for _ in scene['objects']]

    
num_images = len(h5py.File(opt.clevr_img_h5, 'r')['images'])

model.to('cuda')

for data, _, sources, targets, img_ids in tqdm(dataloader, 'processing objects batches'):
    data = data.to('cuda')
    
    preds = model.forward(data)
    
    if not opt.use_proba:
        labels = torch.argmax(preds, dim=1).cpu().numpy()
    else:
        labels = nn.functional.softmax(preds, dim=1).cpu().detach().numpy()
        
    for i, label in enumerate(labels):
        img_id = img_ids[i]
        edge = (sources[i].item(), targets[i].item())

        if not opt.use_proba:
            relation = relation_names[label]
            scenes[img_id]['relationships'][relation][edge[0]].append(edge[1])
        else:
            for rel, proba in enumerate(label):
                relation = relation_names[rel]
                scenes[img_id]['relationships'][relation][edge[0]].append((edge[1], proba.item()))
                
with open(opt.output_path, 'w') as f:
    json.dump({'scenes': scenes}, f)
print('Output scenes saved!')
