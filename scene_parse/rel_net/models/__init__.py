from scene_parse.rel_net.models.rel_net_classification_model import RelNetModule
from scene_parse.rel_net.models.scene_based_model import SceneBasedRelNetModule


def get_model(opt):
    if opt.model_type == 'scene_based':
        model = SceneBasedRelNetModule(opt)
    else:
        model = RelNetModule(opt)
    return model
