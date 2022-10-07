cd ../../
export PYTHONPATH="$PYTHONPATH;$(pwd)"

 echo "attr net 14"
 python scene_parse/attr_net/tools/run_test.py --config_fp measurements/configs/block/attr_nets/attr_net_config_14.yaml
 echo "attr net 24"
 python scene_parse/attr_net/tools/run_test.py --config_fp measurements/configs/block/attr_nets/attr_net_config_24.yaml
 echo "attr net 34"
 python scene_parse/attr_net/tools/run_test.py --config_fp measurements/configs/block/attr_nets/attr_net_config_34.yaml
 echo "attr net 44"
 python scene_parse/attr_net/tools/run_test.py --config_fp measurements/configs/block/attr_nets/attr_net_config_44.yaml

 echo "rel net 14"
 python scene_parse/rel_net/tools/run_test.py --config_fp measurements/configs/block/rel_nets/rel_net_config_14.yaml
 echo "rel net 24"
 python scene_parse/rel_net/tools/run_test.py --config_fp measurements/configs/block/rel_nets/rel_net_config_24.yaml
 echo "rel net 34"
 python scene_parse/rel_net/tools/run_test.py --config_fp measurements/configs/block/rel_nets/rel_net_config_34.yaml
 echo "rel net 44"
 python scene_parse/rel_net/tools/run_test.py --config_fp measurements/configs/block/rel_nets/rel_net_config_44.yaml

echo "solve data_14 rel_scene"
python scene_graph_solver/solve_scenes.py \
    --dataset_name block --folder ./results/clevr_block_val/original/constraint_1234/data_14/\
    --src_file rel_scenes.json\
    --schema_fp data/clevr_block/clevr_attr_map.json\
    --output_file rel_scenes_fix_c1234.json

echo "solve data_24 rel_scene"
python scene_graph_solver/solve_scenes.py \
    --dataset_name block --folder ./results/clevr_block_val/original/constraint_1234/data_24/\
    --src_file rel_scenes.json\
    --schema_fp data/clevr_block/clevr_attr_map.json\
    --output_file rel_scenes_fix_c1234.json

echo "solve data_34 rel_scene"
python scene_graph_solver/solve_scenes.py \
    --dataset_name block --folder ./results/clevr_block_val/original/constraint_1234/data_34/\
    --src_file rel_scenes.json\
    --schema_fp data/clevr_block/clevr_attr_map.json\
    --output_file rel_scenes_fix_c1234.json

echo "solve data_44 rel_scene"
python scene_graph_solver/solve_scenes.py \
    --dataset_name block --folder ./results/clevr_block_val/original/constraint_1234/data_44/\
    --src_file rel_scenes.json\
    --schema_fp data/clevr_block/clevr_attr_map.json\
    --output_file rel_scenes_fix_c1234.json

 echo "attr coord net 14"
 python scene_parse/attr_net/tools/run_test.py --config_fp measurements/configs/block/attr_nets_coord/attr_net_config_coord_14.yaml
 echo "attr coord net 24"
 python scene_parse/attr_net/tools/run_test.py --config_fp measurements/configs/block/attr_nets_coord/attr_net_config_coord_24.yaml
 echo "attr coord net 34"
 python scene_parse/attr_net/tools/run_test.py --config_fp measurements/configs/block/attr_nets_coord/attr_net_config_coord_34.yaml
 echo "attr coord net 44"
 python scene_parse/attr_net/tools/run_test.py --config_fp measurements/configs/block/attr_nets_coord/attr_net_config_coord_44.yaml

echo "solve data_14 coord_scene"
python scene_graph_solver/solve_scenes.py \
    --dataset_name block --folder ./results/clevr_block_val/original/constraint_1234/data_14/\
    --src_file scenes_coord_prob.json\
    --schema_fp data/clevr_block/clevr_attr_map.json\
    --is_coord\
    --output_file rel_scenes_fix_coord.json

echo "solve data_24 coord_scene"
python scene_graph_solver/solve_scenes.py \
    --dataset_name block --folder ./results/clevr_block_val/original/constraint_1234/data_24/\
    --src_file scenes_coord_prob.json\
    --schema_fp data/clevr_block/clevr_attr_map.json\
    --is_coord\
    --output_file rel_scenes_fix_coord.json

echo "solve data_34 coord_scene"
python scene_graph_solver/solve_scenes.py \
    --dataset_name block --folder ./results/clevr_block_val/original/constraint_1234/data_34/\
    --src_file scenes_coord_prob.json\
    --schema_fp data/clevr_block/clevr_attr_map.json\
    --is_coord\
    --output_file rel_scenes_fix_coord.json

echo "solve data_44 coord_scene"
python scene_graph_solver/solve_scenes.py \
    --dataset_name block --folder ./results/clevr_block_val/original/constraint_1234/data_44/\
    --src_file scenes_coord_prob.json\
    --schema_fp data/clevr_block/clevr_attr_map.json\
    --is_coord\
    --output_file rel_scenes_fix_coord.json