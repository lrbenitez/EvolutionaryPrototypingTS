# -*- coding: utf-8 -*-

from .dtw import dtw, fastdtw


def fitness_fastdtw(C, S, vp=0.01):
    """ Function that calculates the fitness of an individual C. To do this, calculate the distance FastDTW between C
    and each serie of the set S.

    :param C: Individual.
    :param S: Set of time series.
    :param vp: Window size. Value in the range (0, 1).

    :return: Tuple of the form (fitness,) where fitness is the fitness of C w.r.t. the set S.
    """
    fitness = 0
    radio = max(1, int(len(S[0])*vp))
    for s in S:
        fitness += fastdtw(s, C, radius=radio)[0]**2
    return fitness,


def fitness_dtw(C, S):
    """ Function that calculates the fitness of an individual C. To do this, calculate the distance DTW between C
    and each serie of the set S.

    :param C: Individual.
    :param S: Set of time series.

    :return: Tuple of the form (fitness,) where fitness is the fitness of C w.r.t. the set S.
    """
    fitness = 0
    for s in S:
        fitness += dtw(s, C)[0]**2
    return fitness,
