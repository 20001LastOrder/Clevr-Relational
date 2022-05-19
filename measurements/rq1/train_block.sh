cd ../../
export PYTHONPATH="$PYTHONPATH;$(pwd)"

echo "attr net 14"
python scene_parse/attr_net/tools/run_train.py --config_fp measurements/rq1/block/attr_nets/attr_net_config_14.yaml
echo "attr net 24"
python scene_parse/attr_net/tools/run_train.py --config_fp measurements/rq1/block/attr_nets/attr_net_config_24.yaml
echo "attr net 34"
python scene_parse/attr_net/tools/run_train.py --config_fp measurements/rq1/block/attr_nets/attr_net_config_34.yaml
# echo "attr net 44"
# python scene_parse/attr_net/tools/run_train.py --config_fp measurements/rq1/block/attr_nets/attr_net_config_44.yaml

echo "attr coord net 14"
python scene_parse/attr_net/tools/run_train.py --config_fp measurements/rq1/block/attr_nets_coord/attr_net_config_coord_14.yaml
echo "attr coord net 24"
python scene_parse/attr_net/tools/run_train.py --config_fp measurements/rq1/block/attr_nets_coord/attr_net_config_coord_24.yaml
echo "attr coord net 34"
python scene_parse/attr_net/tools/run_train.py --config_fp measurements/rq1/block/attr_nets_coord/attr_net_config_coord_34.yaml
# echo "attr coord net 44"
# python scene_parse/attr_net/tools/run_train.py --config_fp measurements/rq1/block/attr_nets_coord/attr_net_config_coord_44.yaml

echo "rel net 14"
python scene_parse/rel_net/tools/run_train.py --config_fp measurements/rq1/block/rel_nets/rel_net_config_14.yaml
echo "rel net 24"
python scene_parse/rel_net/tools/run_train.py --config_fp measurements/rq1/block/rel_nets/rel_net_config_24.yaml
echo "rel net 34"
python scene_parse/rel_net/tools/run_train.py --config_fp measurements/rq1/block/rel_nets/rel_net_config_34.yaml
echo "rel net 44"
python scene_parse/rel_net/tools/run_train.py --config_fp measurements/rq1/block/rel_nets/rel_net_config_44.yaml

echo "clevr attr"
python scene_parse/attr_net/tools/run_train.py --config_fp measurements/rq1/clevr/attr_nets_coord/attr_net_config_coord_14.yaml