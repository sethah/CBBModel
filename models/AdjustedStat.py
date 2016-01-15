import numpy as np
import pandas as pd
from collections import defaultdict

import util
from test.test_RPI import *
from general.DB import DB

class AdjustedStat(object):
    home_factor = 1.014
    available_stats = {'ppp'}
    _default_guesses = {'ppp': 1.0, 'score': 60}

    def __init__(self, stat, num_iterations=10, n_pre=5.):
        """
        Adjusted Stat class.

        An adjusted stat uses an iterative rating method to adjust an offensive
        and defensive version of a statistic for strength of schedule.

        Parameters
        ----------


        Attributes
        -------
        stat : string
            The statistic to adjust.
        num_iterations : int (default=10.)
            Maximum number of iterations for rating algorithm to converge.
        n_pre : int (default=5.)
            The number of games before the preseason effect dies out.
        """
        self.n_pre = n_pre
        self.stat = stat
        self.num_iterations = num_iterations

    def _preseason_rank(self, teams):
        """Compute the preseason rank for each team's offensive and defensive stat."""
        # TODO: Add a preseason rank based on previous end of year rating
        avg_o = 1.
        preseason_o = np.ones(teams.shape[0]) * avg_o
        preseason_d = np.ones(teams.shape[0]) * avg_o
        return preseason_o, preseason_d

    def _initial_guess(self, stacked, teams, gidx):
        """The initial guess before iteration begins."""
        if gidx == 0:
            avg_o = AdjustedStat._default_guesses[self.stat]
        else:
            avg_o = np.mean(stacked[self.stat].values[:gidx])
        guess_o = np.ones(teams.shape[0]) * avg_o
        guess_d = np.ones(teams.shape[0]) * avg_o
        return guess_o, guess_d

    def _average_stats(self, oraw, draw, current_index):
        """The average offensive and defensive stats."""
        s = 0
        c = 0
        for i, team_stats in enumerate(oraw):
            team_index = current_index[i]
            s += np.sum(team_stats[:team_index])
            c += team_index
        avg = float(s) / c
        return avg, avg

    @staticmethod
    def _weights(n, n_pre=5.):
        """
        Initialize data vectors before iteration begins.

        Behavior:
            -Uses uniform weights for the games.
            -Uses a preseason weight that dies out linearly with the number
             of games.

        Parameters
        ----------
        n : int
            The number of game weights to return.
        n_pre : int (default=5.)
            The number of games before the preseason should no longer have an effect.

        Returns
        -------
        w : 1d numpy array
            A vector of game weights.
        w_pre : float
            The preseason rating weight.
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
    def _calc_residual(old, new):
        """Compute the change in values between iterations"""
        return np.linalg.norm(old - new)

    @staticmethod
    def _check_convergence(residual, tol):
        """Check if a stat vector has converged."""
        assert residual > 0, 'residual should be positive'
        return residual < tol

    @staticmethod
    def _is_converged(residual_o, residual_d, tol=0.001):
        """Check if the rate algorithm has converged."""
        return AdjustedStat._check_convergence(residual_o, tol) and \
            AdjustedStat._check_convergence(residual_d, tol)

    def rate(self, stacked, unstacked, teams, game_skip=30, cache_intermediate=False,
             verbose=False):
        """
        Run an adjusted stat model rating for the games data provided.

        By default, provides incremental ratings throughout the season, running
        the rating algorithm every `game_skip` games. For example, if `game_skip`
        is 1, then the algorithm provides update ratings after each game played
        during the season, and there will be `num_games` sets of ratings. The run
        time gets progressively slower as the data included grows throughout the
        season.

        Stats are adjusted according to:
            adj = \sum raw_stat / adj_opp_stat * avg_stat * w_i * \\
                  loc_i + w_pre * stat_pre

        References
        ----------
        TODO
        http://kenpom.com

        Parameters
        ----------
        stacked : dataframe
            Stacked dataframe containing game and stat information.
        unstacked : dataframe
            Unstacked dataframe containing game and stat information. Contains exactly
            half the number of rows as `stacked`.
        teams : dataframe
            Dataframe mapping teams to indices in the data arrays.
        game_skip : int
            The game interval for new ratings. Ratings are computed every `game_skip`
            games. (default=30)
        cache_intermediate : boolean
            Indicating whether to cache the intermediate stat vectors. Mostly
            for debugging. (default=False)
        verbose : boolean
            Log iteration information. (default=False).

        Returns
        -------
        adj_o_history : list[1d numpy array]
            List of adjusted offensive vectors for each run.
        adj_d_history : list[1d numpy array]
            List of adjusted defensive vectors for each run.
        results : list[dictionary]
            List of iteration results for each run of the algorithm.
        """
        # TODO: this needs some refinement but working well for the most part
        idx, loc, oraw, draw = self._initialize(unstacked, teams)
        o_pre, d_pre = self._preseason_rank(teams)

        num_teams = len(idx)

        results = []

        # Add the preseason rank as a starting point
        zero_guess_o, zero_guess_d = self._preseason_rank(teams)
        adj_o_history = [zero_guess_o]
        adj_d_history = [zero_guess_d]

        game_indices = unstacked[['i_hteam', 'i_ateam']].values
        current_index = {team: 0 for team in xrange(num_teams)}
        for gidx, (hidx, aidx) in enumerate(game_indices):
            # increment team vector indices to include new game
            current_index[hidx] += 1
            current_index[aidx] += 1
            if gidx % game_skip != 0:
                continue

            if verbose == True:
                print 'No. of games included: %s' % gidx

            iteration_results = {'o_residual': [], 'd_residual': [], 'iterations': 0}

            avg_o, avg_d = self._average_stats(oraw, draw, current_index)
            adj_o, adj_d = self._initial_guess(stacked, teams, gidx)

            if cache_intermediate:
                iteration_results['o_history'] = []
                iteration_results['d_history'] = []

            for i in xrange(self.num_iterations):
                old_adj_o = adj_o.copy()
                old_adj_d = adj_d.copy()
                for j in xrange(num_teams):
                    k = current_index[j]
                    if k == 0:
                        continue

                    w, w_pre = AdjustedStat._weights(k, self.n_pre)
                    adj_o[j] = AdjustedStat._adjust(oraw[j][:k], adj_d[idx[j][:k]],
                                                    avg_o, w, loc[j][:k], w_pre, o_pre[j])
                    adj_d[j] = AdjustedStat._adjust(draw[j][:k], adj_o[idx[j][:k]],
                                                    avg_d, w, loc[j][:k], w_pre, d_pre[j])
                if cache_intermediate:
                    iteration_results['o_history'].append(adj_o)
                    iteration_results['d_history'].append(adj_d)
                oresidual = AdjustedStat._calc_residual(old_adj_o, adj_o)
                dresidual = AdjustedStat._calc_residual(old_adj_d, adj_d)
                iteration_results['o_residual'].append(oresidual)
                iteration_results['d_residual'].append(dresidual)

                if AdjustedStat._is_converged(oresidual, dresidual):
                    break

            iteration_results['iterations'] = i
            adj_o_history.append(adj_o.copy())
            adj_d_history.append(adj_d.copy())

            results.append(iteration_results)

        return adj_o_history, adj_d_history, results

    @staticmethod
    def _adjust(raw_stat, adj_opp_stat, avg_stat, weights, loc, weight_pre, stat_pre):
        """
        Adjust a statistic to compensate for opponent strength and home court
        for a single team.

        Stats are adjusted according to:
            adj = \sum raw_stat / adj_opp_stat * avg_stat * w_i * loc_i + w_pre * stat_pre

        References
        ----------
        TODO

        Parameters
        ----------
        raw_stat : 1d numpy array
            Vector of raw stat outputs for the team.
        adj_opp_stat : 1d numpy array
            Vector of current values for the team's opponent's adjusted stats.
        avg_stat : float
            The league average for the statistic.
        weights : 1d numpy array
            Vector of weights for individual games.
        loc : 1d numpy array
            Vector of location multipliers.
        weight_pre : float
            Preseason rating weight.
        stat_pre : float
            The preseason stat rating for the team.

        Returns
        -------
        adjusted : float
            The adjusted statistic.
        """
        wsum = np.sum(weights) + weight_pre
        assert np.abs(wsum - 1.0) < 0.001, "weight sum must be 1.0, but was %s" % wsum

        evidence = raw_stat / adj_opp_stat * loc * weights
        prior = weight_pre * stat_pre

        return avg_stat * np.sum(evidence) + prior

if __name__ == "__main__":
    unstacked, stacked, teams = util.get_data(2012)
    all_teams = pd.read_sql("SELECT ncaa, ncaaid FROM teams", DB.conn)
    teams = teams.merge(all_teams, how='left', left_on='team_id', right_on='ncaaid')
    adj = AdjustedStat('ppp', n_pre=1)
    o, d, r = adj.rate(stacked, unstacked, teams, cache_intermediate=True)
    teams['o'] = o[-1]
    teams['d'] = d[-1]
    teams['diff'] = teams.o - teams.d
