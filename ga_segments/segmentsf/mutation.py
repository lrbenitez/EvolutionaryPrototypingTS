# -*- coding: utf-8 -*

import numpy as np
import random
from .interpolation import interpolate
from deap import creator


def mutation(individual, desp, mu, sigma, sigma_extrem):
    """
    Function that mutates an individual. It contains three types of mutation, each of which with a probability. The
    mutations are lateral displacement, smooth vertical displacement and extreme vertical displacement, with
    probabilities of 0.4, 0.4 and 0.2 respectively.

    Lateral displacement: Two segments of the individual are randomly selected. One of them narrows and the other
    widens.

    Smooth vertical offset: A segment is selected and raised or lowered.

    Extreme vertical displacement: A gene of the individual is selected and raised or lowered.

    :param desp: Maximum size of the displacement.
    :param mu: Average of the Gaussian distribution of the soft vertical mutation.
    :param sigma: Standard deviation of the Gaussian distribution of the soft vertical mutation.
    :param sigma_extrem: Standard deviation of the Gaussian distribution of the extreme vertical mutation.

    :return: mutated individual
    """
    d = random.randint(1, desp) * random.choice([-1, 1])
    p = random.random()
    if len(individual) < 3*abs(d):
    	print("Impossible displacement!")
    else:
        if p < 0.4:
            ant = individual[:]
            c1 = random.choice(range(0, len(individual)))
            if len(individual[:c1]) > abs(d) and len(individual[c1+1:]) > abs(d):
                set_c2 = list(range(0, c1-abs(d))) + list(range(c1+abs(d)+1, len(individual)))
                c2 = random.choice(set_c2)
            elif len(individual[:c1]) > abs(d):
                set_c2 = range(abs(d)+1, c1-abs(d))
                c2 = random.choice(set_c2)
            else:
                set_c2 = range(c1+abs(d)+1, len(individual)-abs(d))
                c2 = random.choice(set_c2)

            if c1 > c2:
                c1, c2 = c2, c1

            if len(individual[:c1]) > abs(d) and len(individual[c2:]) > abs(d):
                if random.random() < 0.5:
                    c3 = random.choice(range(0, c1-abs(d)))
                    c4 = random.choice(range(c3+abs(d)+1, c1+1))
                    c1, c2, c3, c4 = c3, c4, c1, c2
                else:
                    c3 = random.choice(range(c2, len(individual)-abs(d)))
                    c4 = random.choice(range(c3+abs(d)+1, len(individual)+1))
            elif len(individual[:c1]) > abs(d):
                c3 = random.choice(range(0, c1-abs(d)))
                c4 = random.choice(range(c3+abs(d)+1, c1+1))
                c1, c2, c3, c4 = c3, c4, c1, c2
            else:
                c3 = random.choice(range(c2, len(individual)-abs(d)))
                c4 = random.choice(range(c3+abs(d)+1, len(individual)+1))

            s1 = individual[:c1]
            s2 = interpolate(individual[c1:c2], len(individual[c1:c2])+d)
            s3 = individual[c2:c3]
            s4 = interpolate(individual[c3:c4], len(individual[c3:c4])-d)
            s5 = individual[c4:]

            individual = s1 + s2 + s3 + s4 + s5

        elif p < 0.4:
            ant = individual[:]
            size = len(individual)
            cxpoint1 = random.randint(1, size)
            cxpoint2 = random.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1

            d = random.gauss(mu, sigma)
            for i in range(cxpoint1, cxpoint2):
                individual[i] = individual[i] + d * (random.random()*0.1 + 0.9)

        else:
            c = random.randint(0, len(individual)-1)
            individual[c] = individual[c] + random.gauss(mu, sigma_extrem)
        return creator.Individual(individual),
        
