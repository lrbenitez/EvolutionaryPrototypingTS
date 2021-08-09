# -*- coding: utf-8 -*-
import time
import multiprocessing

from deap import base
from deap import creator
from deap import tools

from .segmentsf import *

import numpy as np
import random


class GA_segments:
    """Class that contains the function of the genetic algorithm to calculate the center of a set of series using DTW as a
    measure of distance.
    """
    def __init__(self,
                 pop_size=100,
                 ngen=200,
                 cxpb=0.2,
                 mutpb=0.1,
                 mutparams=None,
                 selparams=None,
                 batch_evaluate=False,
                 batch_size=0.1,
                 verbose=False,
                 multi_jobs=False,
                 save_time=False):
        """
        :param pop_size: Population size.
        :param ngen: Generations.
        :param cxpb: Crossover probability.
        :param mutpb: Mutation probability.
        :param batch_evaluate: This parameter indicate if the fitness is calculated with batches or not.
        :param batch_size: Proportion of the dataset used as batch in the fitness evaluation in each generation. Only
        used if batch_evaluate = True. The range of this parameter is (0, 1].
        :param verbose: Verbosity level.
        :param multi_jobs: If True, the evolution process use all cores. If False use one core.
        """
        self.pop_size = pop_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.batch_evaluate = batch_evaluate
        self.batch_size = batch_size
        self.verbose = verbose
        self.multi_jobs = multi_jobs
        self.save_time = save_time

        if mutparams is None:
            self.mutparams = {
                'mu': 0,
                'sigma': 0.03,
                'sigma_extrem': 0.3,
                'desp': 0.04,
            }
        else:
            self.mutparams = mutparams

        if selparams is None:
            self.selparams = {
                'tournsize': 10
            }
        else:
            self.selparams = selparams

    def varAnd(self, population, toolbox):
        """The population is modified by applying, in the first place, the crossing function. Then the mutation is applied.
         The modified population is returned.
        """
        offspring = [toolbox.clone(ind) for ind in population]

        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < self.cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                              offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < self.mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        return offspring

    def ag(self,
           population,
           S,
           toolbox,
           stats=None,
           halloffame=None,
           Stime=None,
           NS=None):
        """General scheme of genetic algorithm. Before starting the evolutionary process, an initial evaluation of the
        individuals is performed to calculate their fitness. If the evaluation is with subsets, a subset of S is
        determined with which all the individuals in this initial evaluation will be evaluated.

        Once the final evaluation is finished, the evolutionary process begins. First, the offspring is selected. Then
        it is modified by the function 'varAnd'. Finally, new individuals are reevaluated (those who have an
        invalid fitness), in case of an evaluation without subsets. In the case of an evaluation with subsets, all
        individuals will be taken as if they had invalid fitness for all to be reevaluated. Once this process is
        finished, the next generation begins.

        :param population: List of individuals.
        :param S: Lista de series de las cuáles se calcula el centroide.
        :param toolbox: Object of the class 'deap.base.Toolbox' that contains the operators of mutation, crossover, etc.
        :param stats: Object of the class 'deap.tools.Statistics' that conatins stats about the evolutionary process.
        :param halloffame: Object of the class 'deap.tools.HallOfFame' that contains the best individuals obtained in
        the evolutionary process.

        :return The final population.
        :return Object of the class 'deap.tools.Logbook' with information about the evolutionary process.
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        if self.save_time:
            step = int(0.05 / self.batch_size)
            if step == 0:
                step = 1
            tmedida = 0
            self.timesg = []

        if self.batch_evaluate:
            batch_n = int(self.batch_size*len(S))
            if batch_n < 1:
                batch_n = 1

        if not self.batch_evaluate:
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            S_selection = S
        else:
            invalid_ind = [ind for ind in population]
            S_selection = random.sample(S, batch_n)

        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind, S_selection)

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if self.verbose:
            print(logbook.stream)

        if self.save_time:
            t1 = time.time()
            ind_mejor = toolbox.selBest(population)[0]
            ind_mejor = NS.desnormalize([ind_mejor])[0]
            fmejor = (fitness.fitness_dtw(ind_mejor, Stime)[0] / len(Stime)) / len(Stime[0])
            self.timesg.append({'time':0,
                                'fitness':fmejor})
            print('[0]', 't:', 0, 'f:', fmejor)

        # Begin the generational process
        for gen in range(1, self.ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = self.varAnd(offspring, toolbox)

            # Evaluate the individuals with an invalid fitness
            if not self.batch_evaluate:
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                S_selection = S
            else:
                invalid_ind = [ind for ind in offspring]
                S_selection = random.sample(S, batch_n)

            # avoids the repetition of evaluations
            individuos_evaluados = []
            for ind in invalid_ind:
                if not ind in individuos_evaluados:
                    ind.fitness.values = toolbox.evaluate(ind, S_selection)
                    individuos_evaluados.append(ind)
                else:
                    ind.fitness.values = individuos_evaluados[individuos_evaluados.index(ind)].fitness.values

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if self.verbose:
                print(logbook.stream)

            if self.save_time and gen % step == 0:
                t2 = time.time()
                ind_mejor = toolbox.selBest(population)[0]
                ind_mejor = NS.desnormalize([ind_mejor])[0]
                fmejor = (fitness.fitness_dtw(ind_mejor, Stime)[0] / len(Stime)) / len(Stime[0])

                tmedida += time.time() - t2
                ttotal = time.time() - t1 - tmedida
                self.timesg.append({'time':ttotal, 'fitness':fmejor})

                print('[{0}]'.format(gen), 't:', ttotal, 'f:', fmejor)

        return population, logbook

    def register_toolbox(self, Sn):
        """Register the operators.
        :param Sn: Normalized series of which the centroid is calculated.

        :return Object of the class:'deap.base.Toolbox'.
        """
        # substitutes the relative value of displacement by the number of elements to be displaced based on the set of
        # given series.
        self.mutparams['desp'] = int(self.mutparams['desp'] * len(Sn[0]))
        if self.mutparams['desp'] < 2:
        	self.mutparams['desp'] = 2

        toolbox = base.Toolbox()
        toolbox.register('generate', generate.sample_generate, S=Sn)
        toolbox.register('population', tools.initRepeat, list, toolbox.generate)
        toolbox.register('evaluate', fitness.fitness_dtw)
        toolbox.register('mutate', mutation.mutation, **self.mutparams)
        toolbox.register('mate', crossover.crossover)
        toolbox.register('select', tools.selTournament, **self.selparams)
        toolbox.register('selBest', tools.selBest, k=1)
        return toolbox

    def calculate_centroids(self, S):
        """Function that calculates the centroid of a set of time series. First register the operators and then the
        evolutionary process is performed.
        :param S: Set of series from which the centroid is calculated.

        :returns: The centroid of S.
        :returns: Centroid fitness.
        :returns: Object of the class: 'deap.tools.Logbook' with information about the evolutionary process.
        """
        if isinstance(S, np.ndarray):
            S = S.tolist()

        NS = normalizacion.Normalize()
        Sn = NS.normalize(S)

        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        toolbox = self.register_toolbox(Sn)

        if self.multi_jobs:
            pool = multiprocessing.Pool()
            toolbox.register('map', pool.map)

        pop = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(3)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        _, log = self.ag(pop, Sn, toolbox, stats=stats, halloffame=hof, Stime=S, NS=NS)

        C = hof[0]
        C = NS.desnormalize([C])[0]
        fitness_mejor = fitness.fitness_dtw(C, S)[0]

        if self.multi_jobs:
            pool.close()

        return C, fitness_mejor, log
