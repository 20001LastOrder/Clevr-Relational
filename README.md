# Clevr-Relational

---
**NOTE**

The appendix with proof for Theorem 1 of the paper, generated scenes and code to run the measurements (Same as the 
one in the `measurements` folder) of the paper can be found [here](https://figshare.com/articles/software/Supplementary_Materials_for_Consistent_Scene_Graph_Generation_By_Constraint_Optimization_/19726882)

---

## Installation
1. install [PyTorch](https://pytorch.org/)==1.10.0 based on your system configuration
2. Install other dependencies with `pip install -r requirements.txt`
3. Get the generated datasets from  [here](https://drive.google.com/file/d/1dolMsZhSFEZNqwoH0j6t3TRN0QEMxXsU/view?usp=sharing)

## Object Detector
1. Train the object detector with (Change the path in object_detector.yaml if needed)
   ```
   python scene_parse/object_detector/train.py --config_fp clevr/object_detector.yaml
   ```
2. Generate object proposals with 
    ```
      python scene_parse/object_detector/predict.py\ 
       --dataset_name <clevr or block>\
       --weight_path <path to model weight>\
       --image_h5 <path to image file>\
       --output_fp <output path>\
       --num_categories 1\
       --score_threshold 0.5
    ```

3. Process the proposal to generate object only scenes
   ```
   python scene_parse/attr_net/tools/process_proposals.py\
        --attribute_map <path to attr_map.json>\
        --gt_scene_path <path to ground truth scene> (Remove this option for test scenes)\
        --proposal_path <output file from 2>\
        --score_thresh 0.5\
        --output_path <output path>\
        --suppression 1
   ```
4. Repeat 2 and 3 for the test images


## Train attribute detector, relationship detector and running scene fixing
It is easiest to check sample scripts and configuration to run the training in `measurements/configs`. It follows the following steps
1. Create the configuration for attribute and relationship detector e.g. [attribute config](clevr/attr_net_config.yaml) and [relationship config](clevr/rel_net_config.yaml)
2. Train 
   ```
   python scene_parse/attr_net/tools/run_train.py --config_fp <path_to_attr_config>
   python scene_parse/rel_net/tools/run_train.py --config_fp <path_to_rel_config>
   ```
3. Generate scenes
   ```
   python scene_parse/attr_net/tools/run_test.py --config_fp <path_to_attr_config>
   python scene_parse/rel_net/tools/run_test.py --config_fp <path_to_rel_config>
   ```
4. Fixing scenes with constraints (Note, you'll probably need to download and config [Gurobi](https://www.gurobi.com/) and setup gurobypy accordingly)
   ```
   python scene_graph_solver/solve_scenes.py \
    --dataset_name <clevr or block> --folder <folder containing the scene file>\
    --src_file <scene file name>\
    --schema_fp <path to the schema file>\
    --output_file <output file name>
   ```
## Credit
Part of the code is adopted from [this repo](https://github.com/kexinyi/ns-vqa) 