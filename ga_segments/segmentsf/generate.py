# -*- coding: utf-8 -*-

from deap import creator
import random


def minimo(S):
    return min(map(min, S))


def maximo(S):
    return max(map(max, S))


def random_generate(L, S):
    """
    Function that randomly generates an individual of length L .
    :param L: Length of the individual to be generated.
    :param S: Set of series from which you want to calculate the centroid.

    :return: Individual.
    """
    ind = []
    mini, maxi = minimo(S), maximo(S)
    for i in range(L):
        ind.append(random.random() * (maxi - mini)  + mini)
    return creator.Individual(ind)


def sample_generate(S):
    """
    Function that randomly selects a series from the set S.

    :param S: Set of series from which you want to calculate the centroid..

    :return: Individual.
    """
    return creator.Individual(random.choice(S))
