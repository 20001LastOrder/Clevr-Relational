# Clevr-Relational

## Execute the relationship extractor
1. install [Pytorch Lightning](https://www.pytorchlightning.ai/), [PyTorch](https://pytorch.org/), [PyYaml](https://pypi.org/project/PyYAML/)
    and [h5py](https://www.h5py.org/)
2. Get data from  [here](https://drive.google.com/drive/folders/1-J1AnYBBx8vNkFsWo89ZbdIsRZLcAv-r?usp=sharing)
3. Edit the `test configuration` of `clevr/rel_net_config.yaml` to point to necessary data
    ``` yaml
   test_ann_path: <path to proposals.json>
    test_img_h5: <path to images.h5>
    label_names: ["left", "right", "front", "behind"]
    model_path: <path to model.ckpt>
    scenes_path: <path to attr_scene.json>
    use_proba: False
    output_path: <output file path>
   ```
4. From the root level of the project, run `python scene_parse\rel_net\tools\run_test.py --config_fp clevr/rel_net_config.yaml`