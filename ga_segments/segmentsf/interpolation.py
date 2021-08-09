# -*- coding: utf-8 -*-

import numpy as np


def interpolate(x, m):
    """ Function that obtains a series of length m from the serie x by linear interpolation.

    :param x: serie.
    :param m: output length of the serie.

    :return: interpolated serie of length m.
    """
    if len(x) == 1:
        return [x[0]] * m
    n = len(x) - 1
    step = n / float(m)
    xs = np.arange(step / 2.0, n, step)
    return np.interp(xs, range(len(x)), x).tolist()
