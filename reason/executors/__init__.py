from .clevr_executor import ClevrExecutor


def get_executor(opt):
    print('| creating %s executor' % opt.dataset)
    if opt.dataset == 'clevr':
        scene_json = opt.clevr_scene_path
        vocab_json = opt.clevr_vocab_path
    else:
        raise ValueError('Invalid dataset')
    executor = ClevrExecutor(scene_json, vocab_json)
    return executor