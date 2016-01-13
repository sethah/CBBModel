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
        """Get a stacked and unstacked dataframe of game and stat information."""
        self.unstacked, self.stacked, self.teams = util.get_data(time_range)

    def _preseason_rank(self):
        """Compute the preseason rank for each team's offensive and defensive stat."""
        avg_o = 1.
        preseason_o = np.ones(self.teams.shape[0]) * avg_o
        preseason_d = np.ones(self.teams.shape[0]) * avg_o
        return preseason_o, preseason_d

    def _initial_guess(self):
        """The initial guess before iteration begins."""
        avg_o = self.stacked[self.stat].mean()
        guess_o = np.ones(self.teams.shape[0]) * avg_o
        guess_d = np.ones(self.teams.shape[0]) * avg_o
        return guess_o, guess_d

    @staticmethod
    def _weights(n, n_pre=5.):
        """
        Get the weight vector for `n` games played.
        :param n: INT - number of games played.
        :param n_pre: INT - number of games until preseason stops having an effect.
        :return: 1D Numpy Array, FLOAT
        """
        assert n > 0, 'n must be a positive integer'
        if n == 0:
            return np.array([]), 1.
        n = int(n)
        w = np.ones(n)
        w_pre = max(0, 1. - n / n_pre)
        w_norm = 1 - w_pre
        w /= (w.sum() / w_norm)
        return w, w_pre

    @staticmethod
    def _loc_factor(is_home, is_neutral):
        """The location multiplier for a game."""
        if is_neutral:
            loc_factor = 1.
        elif is_home:
            loc_factor = AdjustedStat.home_factor
        else:
            loc_factor = 1. / AdjustedStat.home_factor
        return loc_factor

    def _initialize(self, unstacked, teams):
        """
        Initialize data vectors before iteration begins.

        Parameters
        ----------
        unstacked : dataframe
            Unstacked dataframe containing game and stat information.
        teams : dataframe
            Dataframe mapping teams to indices in the data arrays.

        Returns
        -------
        idx : list[1d numpy array]
            List whose elements are vectors of opponent indices for each team.
        loc : list[1d numpy array]
            List whose elements are vectors of location multipliers for each team.
        offense : list[1d numpy array]
            List whose elements are vectors of raw offensive outputs for each team.
        defense : list[1d numpy array]
            List whose elements are vectors of raw defensive outputs for each team.

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
            arr = [[] for i in xrange(teams.shape[0])]
            for idx, vals in d.iteritems():
                arr[idx] = np.array(vals)
            return arr

        offense = _convert_dict(odict)
        defense = _convert_dict(ddict)
        idx = _convert_dict(idx_dict)
        loc = _convert_dict(loc_dict)

        return idx, loc, offense, defense

    @staticmethod
    def _update_residual(old, new):
        """Compute the change in values between iterations"""
        return np.linalg.norm(old - new)

    def rate(self, time_range):
        """
        Run an adjusted stat model rating for the games within the
        specified time range.

        Stats are adjusted according to:
            adj = \sum raw_stat / adj_opp_stat * avg_stat * w_i * loc_i + w_pre * stat_pre

        References
        ----------
        http://kenpom.com

        Parameters
        ----------
        time_range : date_like
            A date or year representing the time range of interest.

        Returns
        -------
        adj_o : 1d numpy array
            vector of adjusted offensive stats for each team.
        adj_d : 1d numpy array
            vector of adjusted defensive stats for each team.
        residual : 1d numpy array
            Vector of residuals at each iteration.
        """
        self._get_data(time_range)
        idx, loc, oraw, draw = self._initialize(self.unstacked, self.teams)
        o_pre, d_pre = self._preseason_rank()

        num_teams = len(idx)
        adj_o, adj_d = self._initial_guess()
        residual = {'offense': [], 'defense': []}
        for i in xrange(self.num_iterations):
            old_adj_o = adj_o.copy()
            old_adj_d = adj_d.copy()
            for j in xrange(num_teams):
                num_games = idx[j].shape[0]
                w, w_pre = AdjustedStat._weights(num_games, self.n_pre)

                new_o = np.sum(oraw[j] / adj_d[idx[j]] * w * loc[j] + w_pre * o_pre[j])
                new_d = np.sum(draw[j] / adj_o[idx[j]] * w * (1. / loc[j]) + w_pre * d_pre[j])

                adj_o[j] = new_o
                adj_d[j] = new_d
            residual['offense'].append(AdjustedStat._update_residual(old_adj_o, adj_o))
            residual['defense'].append(AdjustedStat._update_residual(old_adj_d, adj_d))

        return adj_o, adj_d, residual

if __name__ == "__main__":
    adj = AdjustedStat('ppp')
    o, d, r = adj.rate(2012)
