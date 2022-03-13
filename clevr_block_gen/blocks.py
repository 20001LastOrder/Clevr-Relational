import json
import random
import numpy as np

properties = {}


def load_colors(args):
    # Load the property file
    with open(args.properties_json, 'r') as f:
        properties.update(json.load(f))

        # changes color value range from 0-255 to 0-1
        properties["colors"] = {name:
                                    tuple(float(c) / 255.0 for c in rgb) + (1.0,) \
                                for name, rgb in properties['colors'].items()
                                }

        # extract exactly the same numbr of colors as the objects
        # from the top in the order as written in the json file
        # properties["colors"] = properties["colors"][:args.max_objects]

    return properties


def random_dict(dict):
    return random.choice(list(dict.items()))


class Unstackable(Exception):
    pass


class Block(object):
    def __init__(self, i):
        shape_name, _ = random_dict(properties['shapes'])
        self.shape = shape_name
        self.color, _ = random_dict(properties['colors'])
        self.size, _ = random_dict(properties['sizes'])
        self.material, _ = random_dict(properties['materials'])
        self.rotation = 360.0 * random.random()
        self.stackable = properties['stackable'][shape_name] == 1
        self.location = [0, 0, 0]
        self.id = i
        pass

    @property
    def x(self):
        return self.location[0]

    @property
    def y(self):
        return self.location[1]

    @property
    def z(self):
        return self.location[2]

    @x.setter
    def x(self, newvalue):
        self.location[0] = newvalue

    @y.setter
    def y(self, newvalue):
        self.location[1] = newvalue

    @z.setter
    def z(self, newvalue):
        self.location[2] = newvalue

    def __eq__(o1, o2):
        if o1 is None:
            return False
        if o2 is None:
            return False
        return o1.id == o2.id

    def similar(o1, o2):
        if o1 is None:
            return False
        if o2 is None:
            return False
        return \
            o1.color == o2.color and \
            o1.size == o2.size and \
            o1.material == o2.material

    def left(self, o2):
        return not self.overlap(o2) and self.x < o2.x

    def overlap(self, o2):
        return abs(self.x - o2.x) < (properties['sizes'][self.size] + properties['sizes'][o2.size])

    def stable_on(self, o2):
        return abs(self.x - o2.x) < properties['sizes'][o2.size]

    def above(self, o2):
        return self.overlap(o2) and (self.z > o2.z)


class State(object):
    "Randomly select a list of objects while avoiding duplicates"

    def __init__(self, args):
        objects = []
        num_objects = random.randrange(args.min_objects, args.max_objects + 1)
        for i in range(num_objects):
            while True:
                o1 = Block(i)
                if args.allow_duplicates:
                    break
                ok = True
                for o2 in objects:
                    if o1.similar(o2):
                        ok = False
                        print("duplicate object!")
                        break
                if ok:
                    break
            objects.append(o1)

        self.table_size = args.table_size
        self.object_jitter = args.object_jitter
        self.objects = objects
        self.shuffle()
        pass

    def for_rendering(self):
        return [vars(o) for o in sorted(self.objects, key=(lambda o: o.id))]

    def shuffle(self):
        """destructively modify the list of objects using shuffle1."""
        objs = self.objects.copy()
        self.objects.clear()
        for oi in objs:
            self.shuffle1(oi)
            self.objects.append(oi)

    def shuffle1(self, oi, force_change=False):
        """destructively modify an object by choosing a random x position and put it on top of existing objects.
     oi itself is not inserted to the list of objects."""
        # note: if a cube is rotated by 45degree, it should consume 1.41 times the size
        unit = max(properties['sizes'].values())
        max_x = unit * 2 * self.table_size

        if force_change:
            object_below = self.object_just_below(oi)

        trial = 0
        fail = True
        while fail and trial < 100:
            fail = False
            oi.x = max_x * ((random.randint(0, self.table_size - 1) / (self.table_size - 1)) - 1 / 2) + random.gauss(
                0.0, self.object_jitter * unit)
            oi.z = 0
            for oj in self.objects:
                if oi.overlap(oj):
                    if not oj.stackable:
                        fail = True
                        break
                    if not oi.stable_on(oj):
                        fail = True
                        break
                    oi.z = max(oi.z, oj.z + properties['sizes'][oj.size])
            oi.z += properties['sizes'][oi.size]
            if force_change:
                new_object_below = self.object_just_below(oi)
                if object_below == new_object_below:
                    # is not shuffled!
                    fail = True
            trial += 1

        if fail:
            raise Unstackable("this state is not stackable")
        pass

    def tops(self):
        """returns a list of objects on which nothing is on top of, i.e., it is the top object of the tower."""
        tops = []
        for o1 in self.objects:
            top = True
            for o2 in self.objects:
                if o1 == o2:
                    continue
                if o2.above(o1):
                    top = False
                    break
            if top:
                tops.append(o1)
        return tops

    def is_right_most(self, o):
        for other in self.objects:
            if o == other:
                continue
            if o.left(other):
                return False
        return True

    def right_most_objects(self):
        return [o for o in self.objects if self.is_right_most(o)]

    def is_bottom(self, o):
        for other in self.objects:
            if o == other:
                continue
            if o.above(other):
                return False
        return True

    def is_direct_next(self, o1, o2):
        left = min(o1.x, o2.x)
        right = max(o1.x, o2.x)

        for o in self.objects:
            if np.isclose(o.x, left) or np.isclose(o.x, right):
                continue
            if left < o1.x < right:
                return False

        return True

    def objects_above(self, o):
        results = []
        for other in self.objects:
            if o != other and other.above(o):
                results.append(other)
        return results

    def objects_below(self, o):
        results = []
        for other in self.objects:
            if o != other and o.above(other):
                results.append(other)
        return results

    def object_just_below(self, o):
        objects_below = self.objects_below(o)
        if len(objects_below) == 0:
            return None
        else:
            result = objects_below[0]
            for other in objects_below[1:]:
                if result.z < other.z:
                    result = other
            return result

    def random_action(self):
        method = random.choice([self.action_move])
        method()
        # storing the name of the action. This is visible in the json file
        self.last_action = method.__name__
        pass

    def action_move(self):
        o = random.choice(self.tops())
        index = self.objects.index(o)
        self.objects.remove(o)
        self.shuffle1(o, force_change=True)
        self.objects.insert(index, o)
        # note: do not change the order of the object.
        pass

    def action_change_material(self):
        o = random.choice(self.tops())
        tmp = list(properties['materials'].values())
        tmp.remove(o.material)
        o.material = random.choice(tmp)
        pass
