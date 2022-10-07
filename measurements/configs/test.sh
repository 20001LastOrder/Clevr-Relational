cd ../../
export PYTHONPATH="$PYTHONPATH;$(pwd)"

echo "attr net test"
python scene_parse/attr_net/tools/run_test.py --config_fp measurements/configs/block/attr_nets/attr_net_config_test.yaml

echo "rel net test"
python scene_parse/rel_net/tools/run_test.py --config_fp measurements/configs/block/rel_nets/rel_net_config_test.yaml

echo "attr coord net test"
python scene_parse/attr_net/tools/run_test.py --config_fp measurements/configs/block/attr_nets_coord/attr_net_config_coord_test.yaml
