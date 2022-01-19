import numpy as np
from numpy.linalg import inv
import json


def get_transformation_matrix(coords):
    dim = len(coords)
    mat = np.identity(dim + 1)
    mat[:-1, -1] += coords
    return mat


def get_rotation_matrix_z(rotation):
    """
    Get the rotation matrix on the z axis in three dimensional space
    http://www.inf.ed.ac.uk/teaching/courses/cg/lectures/cg3_2013.pdf
    :param rotation: rotation on the z axis in degrees
    :return: a matrix perform the rotation transformation
    """
    rad = np.radians(rotation)
    mat = np.identity(4)
    mat[:2, :2] = np.matrix([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return mat


def get_gtl_transformer(coords, rotation):
    """
    Get a transformation function that transform global coordinate to local coordinate of the provided axis system
    :param coords: coordinates
    :param rotation: rotation on the z axis in degree
    :return: a function that transforms global coordinate to local coordinate of the provided axis
    """
    mat = get_rotation_matrix_z(rotation) @ get_transformation_matrix(coords)
    inverse = inv(mat)
    return lambda c: (inverse @ np.array(np.concatenate((c, [1]))))[:len(c)]


def left_rel(obj1, obj2):
    return obj1[1] < obj2[1]


def right_rel(obj1, obj2):
    return obj1[1] > obj2[1]


def before_rel(obj1, obj2):
    return obj1[0] < obj2[0]


def behind_rel(obj1, obj2):
    return obj1[0] > obj2[0]


def read_json(fname):
    with open(fname) as f:
        return json.load(f)


def write_json(obj, fp):
    with open(fp, 'w') as f:
        json.dump(obj, f)
