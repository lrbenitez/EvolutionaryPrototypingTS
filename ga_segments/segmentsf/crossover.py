# -*- coding: utf-8 -*-

from .dtw import dtw
from .interpolation import interpolate
import random


def crossover(G1, G2):
    """Function that crosses two individuals. First, the alignment segments are calculated. Then, it is randomly chosen
    which sequence of segments will be crossed. Finally they are exchanged interpolating the crossed parts so that the
    resulting individuals have the same length as their parents.

    :return: Descendant 1.
    :return: Descendant 2.
    """

    _, W = dtw(G1, G2)
    W1 = W[0]
    W2 = W[1]
    PCs, pc = [], [0]
    for i in range(1, len(W1)-1):

        if W1[i] != W1[i-1] and W2[i] != W2[i-1] and (W1[i] == W1[i+1] or W2[i] == W2[i+1]):
            if pc:
                PCs.append(pc)
            pc = [i]

        elif W1[i] != W1[i+1] and W2[i] != W2[i+1] and (W1[i] == W1[i-1] or W2[i] == W2[i-1]):
            PCs.append(pc)
            pc = [i]
        else:
            pc.append(i)
    PCs.append(pc + [len(W1)-1])

    c1 = random.randint(0, len(PCs)-1)
    c2 = random.randint(0, len(PCs)-1)
    if c1 > c2:
        c1, c2 = c2, c1

    pi, pf = PCs[c1][0], PCs[c2][-1]
    pi1, pf1 = W1[pi], W1[pf]+1
    pi2, pf2 = W2[pi], W2[pf]+1

    G1[pi1:pf1], G2[pi2:pf2] = interpolate(G2[pi2:pf2], len(G1[pi1:pf1])), interpolate(G1[pi1:pf1], len(G2[pi2:pf2]))

    return G1, G2
