import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict

from general.DB import DB
import util

class AdjustedStat(object):
    home_factor = 1.014
    available_stats = {'ppp'}

    def __init__(self, stat, num_iterations=10, n_pre=5.):
        self.n_pre = n_pre
        self.stat = stat
        self.num_iterations = num_iterations

    def _get_data(self, time_range):
        """
        Get and store game and team data for the given time range.
        :param time_range: A year, date, or string representing the time range of interest.
        :return: None
        """
        self.unstacked, self.stacked, self.teams = util.get_data(time_range)

    def _preseason_rank(self):
        """
        Compute the preseason rank for each team's offensive and defensive stat.
        :return: (1D Numpy Array, 1D Numpy Array)
        """
        avg_o = 1.
        preseason_o = np.ones(self.teams.shape[0]) * avg_o
        preseason_d = np.ones(self.teams.shape[0]) * avg_o
        return preseason_o, preseason_d

    def _initial_guess(self):
        avg_o = self.stacked[self.stat].mean()
        preseason_o = np.ones(self.teams.shape[0]) * avg_o
        preseason_d = np.ones(self.teams.shape[0]) * avg_o
        return preseason_o, preseason_d

    @staticmethod
    def _weights(n, n_pre=5.):
        """
        Get the weight vector for `n` games played.
        :param n: INT - number of games played.
        :param n_pre: INT - number of games until preseason stops having an effect.
        :return: 1D Numpy Array, FLOAT
        """
        if n == 0:
            return np.array([]), 1.
        w = np.ones(n)
        w_pre = max(0, 1. - n / n_pre)
        w_norm = 1 - w_pre
        w /= (w.sum() / w_norm)
        return w, w_pre

    @staticmethod
    def _loc_factor(is_home, is_neutral):
        """
        Get the location factor for a game.
        :param is_home: BOOLEAN - whether the team was home or not.
        :param is_neutral: BOOLEAN - indicating a neutral site game.
        :return: FLOAT
        """
        if is_neutral:
            loc_factor = 1.
        elif is_home:
            loc_factor = AdjustedStat.home_factor
        else:
            loc_factor = 1. / AdjustedStat.home_factor
        return loc_factor

    def _initialize(self, unstacked, teams):
        """
        Create arrays of offense and defense raw outputs for the games.
        :return:
        """
        odict = defaultdict(list)
        ddict = defaultdict(list)
        idx_dict = defaultdict(list)
        loc_dict = defaultdict(list)
        cols = ['i_hteam', 'i_ateam', 'h' + self.stat, 'a' + self.stat, 'neutral']
        for row in unstacked[cols].values:
            odict[row[0]].append(row[2])
            ddict[row[0]].append(row[3])
            odict[row[1]].append(row[3])
            ddict[row[1]].append(row[2])
            idx_dict[row[0]].append(row[1])
            idx_dict[row[1]].append(row[0])
            loc_dict[row[0]].append(AdjustedStat._loc_factor(True, row[4]))
            loc_dict[row[1]].append(AdjustedStat._loc_factor(False, row[4]))

        def _convert_dict(d):
            arr = [[] for i in range(teams.shape[0])]
            for idx, vals in d.iteritems():
                arr[idx] = np.array(vals)
            return arr

        offense = _convert_dict(odict)
        defense = _convert_dict(ddict)
        idx = _convert_dict(idx_dict)
        loc = _convert_dict(loc_dict)

        return idx, loc, offense, defense

    def rate(self, time_range):
        self._get_data(time_range)
        idx, loc, oraw, draw = self._initialize(self.unstacked, self.teams)
        o_pre, d_pre = self._preseason_rank()

        adj_o, adj_d = self._initial_guess()

        for i in xrange(self.num_iterations):
            adj_o_prev = adj_o
            for j in xrange(len(idx)):
                w = AdjustedStat._weights(oraw[j].shape[0], self.n_pre)
                new_o = oraw[j] / adj_d[idx[j]] * w * loc[j]
                new_d = draw[j] / adj_o[idx[j]] * w * (1. / loc[j])

if __name__ == "__main__":
    adj = AdjustedStat('score')
    adj._get_data(2012)
