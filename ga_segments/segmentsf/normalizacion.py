# -*- coding: utf-8 -*-
import numpy as np


class Normalize:
    """ Class that normalizes and denormalizes sets of time series. For standardization, the mean and standard deviation
     of all elements of all series are used.
    """
    def normalize(self, S):
        """
        Function that normalizes a set of time series.
        :param S: Time series set.
        :return: Normalized time series set.
        """
        elements = []
        Sn = []

        for s in S:
            elements.extend(s)

        self.media = np.mean(elements)
        self.std = np.std(elements)

        for s in S:
            sn = []
            for i in range(len(s)):
                sn.append((self.media - s[i]) / self.std)
            Sn.append(sn)

        return Sn

    def desnormalize(self, Sn):
        """
        Function that denormalizes a set of time series, resulting in the original time series.
        :param Sn: Normalized time series set.
        :return: Unnormalized time series set.
        """
        S = []
        for sn in Sn:
            s = []
            for i in range(len(sn)):
                s.append(self.media - self.std * sn[i])
            S.append(s)
        return S
