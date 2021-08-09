# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from .ga import GA_segments
from .ga_coop import GA_segments_coop
from .segmentsf import dtw


class NC:
    """ Class that contains the Nearest Centroid algorithm, which classifies a set of time series according to the
    centroid of the class closest to each series.
    """

    def __init__(self, ga='simple', params_ga={}, verbose=0):
        """
        :param ga: The genetic algorithm that is used. It can be 'simple' or 'coop'.
        :param params_ga: Parameters of genetic algorithm.
        :param verbose
        """
        self.verbose = verbose
        self.ga = ga
        self.params_ga = params_ga

    def fit(self, X, y):
        """ Function that calculates the centroids of each class.

        :param X: Series.
        :param y: Class to which each series belongs.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        self.classes = np.unique(y)
        self.centroids = []
        inertia_total = 0

        for i, c in enumerate(self.classes):
            if self.ga == 'simple':
                GA = GA_segments(**self.params_ga)
            elif self.ga == 'coop':
                GA = GA_segments_coop(**self.params_ga)

            mask = y == c
            S = X[mask]
            centroid, inertia, _ = GA.calculate_centroids(S=S)
            self.centroids.append(centroid)

            if self.verbose > 1:
                self._plot(i, centroid)
            if self.verbose > 0:
                self._print(i, inertia)

            inertia_total += inertia
            self.inertia = inertia_total


    def predict(self, X):
        """ Function that calculates the class to which each series of a set of series X belongs. Each series is
        assigned the class of the nearest centroid.
    
        :param X
        """
        if not self.centroids:
            raise Exception('Error: Fit the data first')
    
        if not isinstance(X, np.ndarray):
            X = np.array(X)
    
        self.labels = np.zeros(len(X))
        for i, x in enumerate(X):
            dists = [dtw.dtw(x, c)[0] for c in self.centroids]
            class_index = np.argmin(dists)
            self.labels[i] = self.classes[class_index]
    
    
    def fit_predict(self, X_train, y, X_test):
        """
        Función que calcula los centroides y posteriormente clasifica un conjunto de series. Para ello, llama en primer
        lugar a :meth:`fit` y después a :meth:`predict`.
    
        :param X_train
        :param y_train
        :param X_test
    
        :return: Class of each series of X_test
        """
        self.fit(X_train, y)
        return self.predict(X_test)
    
    
    def pretrained_predict(self, C, X):
        """ Function that classifies the series X given the centroids C. Therefore, in this case the algorithm
        does not need training.
    
        :param C: Centroids of each class.
        :param X: Temporal series.
    
        :return: Class of each series of X.
        """
        self.labels = np.zeros(len(X))
    
        for i, x in enumerate(X):
            min_d = np.inf
            for j, c in enumerate(C):
                d = dtw.dtw(x, c)[0]
                if d < min_d:
                    min_d = d
                    self.labels[i] = j
    
        return self.labels
    
    
    def fuzzy_predict(self, X):
        """ Function that classifies the X series in a diffuse way. The result will be the degree of belief of
        belonging to each class.
    
        :param X: Temporal series.
    
        :return: Probability of belonging to each class.
        """
        if not self.centroids:
            raise Exception('Error: Fit the data first')
    
        if not isinstance(X, np.ndarray):
            X = np.array(X)
    
        self.fuzzy_labels = []
        for x in X:
            dists = [dtw.dtw(x, c)[0] for c in self.centroids]
            total = sum(dists)
            creencias_x = [1 - d / total for d in dists]
            total_creen = sum(creencias_x)
            creencias_x_norm = [cree / total_creen for cree in creencias_x]
            self.fuzzy_labels.append(creencias_x_norm)
        self.fuzzy_labels = np.array(self.fuzzy_labels)
    
    
    def fit_fuzzy_predict(self, X_train, y, X_test):
        """ Function that calculates the centroids and later classifies a set of series in a fuzzy way.
    
        :param X_train
        :param y_train
        :param X_test
    
        :return: Probability of belonging to each class.
        """
        self.fit(X_train, y)
        return self.fuzzy_predict(X_test)
    
    
    def pretrained_fuzzy_predict(self, C, X):
        """ Function that makes a fuzzy classification of the series X given the centroids C. Therefore, in
        this case the algorithm does not need training.
    
        :param C: Centroids of each class.
        :param X: Temporal series.
    
        :return: Probability of belonging to each class.
        """
        self.fuzzy_labels = []
        for x in X:
            dists = [dtw.dtw(x, c)[0] for c in C]
            total = sum(dists)
            creencias_x = [1 - d / total for d in dists]
            total_creen = sum(creencias_x)
            creencias_x_norm = [cree / total_creen for cree in creencias_x]
            self.fuzzy_labels.append(creencias_x_norm)
        self.fuzzy_labels = np.array(self.fuzzy_labels)
    
        return self.fuzzy_labels
    
    
    def score(self, X_test, y):
        """ Function that calculates the accuracy of the classification.
    
        :param X_test
        :param y
    
        :return: Accuracy.
        """
        self.predict(X_test)
        return (self.labels == y).sum() / float(y.shape[0])
    
    
    def _print(self, n, inertia):
        print('Error class[{}]: {:.3f}'.format(n, inertia))
        print('-' * 25)
    
    
    def _plot(self, n, c):
        plt.plot(c)
        plt.title('Centroid {}'.format(n))
        plt.show()
