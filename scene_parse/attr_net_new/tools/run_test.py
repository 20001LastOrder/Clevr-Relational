import sys
import torch
import numpy as np
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '.')
from options import get_options
from datasets import get_test_dataloader
from model import get_model, AttrNetClassificationModule
from trainer import get_trainer
from pytorch_lightning import Trainer
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint
import h5py
from tqdm import tqdm
import json

opt = get_options('test')
model = AttrNetClassificationModule.load_from_checkpoint(opt.model_path)
dataloader = get_test_dataloader(opt)

with open(opt.attr_map_path, 'r') as f:
    attr_map = json.load(f)

attr_names = opt.attr_names    
    
num_images = len(h5py.File(opt.clevr_img_h5, 'r')['images'])

scenes = [{
    'image_index': i,
    'objects': []
} for i in range(num_images)]

model.to('cuda')

for data, _, idxs, img_ids in tqdm(dataloader, 'processing objects batches'):
    data = data.to('cuda')
    
    preds = model.forward(data)
    attribute_labels = np.stack([torch.argmax(pred, dim=1).cpu() for pred in preds]).transpose((1, 0))
    for i, attributes in enumerate(attribute_labels):
        img_id = img_ids[i]
        idx = idxs[i]
        obj = {}
        
        for j, attribute in enumerate(attributes):
            attr_name = attr_names[j]
            obj[attr_name] = attr_map[attr_name][attribute]
            
        scenes[img_id]['objects'].append(obj)
        

with open(opt.output_path, 'w') as f:
    json.dump({'scenes': scenes}, f)
print('Output scenes saved!')
