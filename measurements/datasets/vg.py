from clevr_block_gen.constraints import SceneConstraint

def get_vg_constraint_map():
    return {
        'opposite': VGOppositeConstraint(),
        'no_loop': VGLoopConstraint(),
        'label': VGLabelConstraint(),
    }

class VGOppositeConstraint(SceneConstraint):
    name = 'opposite'
    def check_oppsite(self, rels):
        count = 0
        for source, targets in enumerate(rels):
            for target in targets:
                if source in rels[target]:
                    count += 1
        return count

    def evaluate(self, scene):
        rels = scene['relationships']
        return self.check_oppsite(rels['behind']) + self.check_oppsite(rels['in']) + self.check_oppsite(rels['above']) \
               + self.check_oppsite(rels['under'])


class VGLoopConstraint(SceneConstraint):
    name = 'transitivity'
    def check_transitivity(self, rels):
        count = 0
        for s, targets in enumerate(rels):
            for t in targets:
                for k in rels[t]:
                    if s in rels[k]:
                        count += 1
        return count

    def evaluate(self, scene):
        rels = scene['relationships']
        return self.check_transitivity(rels['behind']) + self.check_transitivity(rels['in']) + \
               self.check_transitivity(rels['above']) + self.check_transitivity(rels['under'])


class VGLabelConstraint(SceneConstraint):
    name='label'
    labels = {'man', 'woman', 'person'}

    def evaluate(self, scene):
        objs = scene['objects']
        for obj in objs:
            if obj['label'] in self.labels:
                return 0
        return 1
