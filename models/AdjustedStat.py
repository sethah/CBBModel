import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.stats as scs

import util
import org_ncaa
from RatingsModel import RatingsModel

class AdjustedStat(RatingsModel):
    """
    Adjusted Stat class.

    An adjusted stat uses an iterative rating method to adjust an offensive
    and defensive version of a statistic for strength of schedule.

    Attributes
    -------
    stat : string
        The statistic to adjust.
    num_iterations : int (default=10)
        Maximum number of iterations for rating algorithm to converge.
    n_pre : int (default=5)
        The number of games before the preseason effect dies out.
    """
    home_factor = 1.014
    pythag_exponent = 11.5
    available_stats = {'ppp'}
    _default_guesses = {'ppp': 1.0, 'score': 60}

    def __init__(self, stat, **kwargs):
        self.params = {}
        self.set_params(**self._default_params())
        self.set_params(stat=stat)
        self.set_params(**kwargs)

    def _default_params(self):
        return {'n_pre': 5, 'stat': None, 'num_iterations': 10, 'game_skip': 30,
                'cache_intermediate': False, 'verbose': False}

    def _preseason_rank(self, teams):
        """Compute the preseason rank for each team's offensive and defensive stat."""
        # TODO: Add a preseason rank based on previous end of year rating
        avg_o = 1.0
        preseason_o = np.ones(teams.shape[0]) * avg_o
        preseason_d = np.ones(teams.shape[0]) * avg_o
        return preseason_o, preseason_d

    def _initial_guess(self, unstacked, teams, gidx=0):
        """The initial guess before iteration begins."""
        if gidx == 0:
            avg_o = AdjustedStat._default_guesses[self.stat]
        else:
            sum_h = np.sum(unstacked['h' + self.stat].values[:gidx])
            sum_a = np.sum(unstacked['a' + self.stat].values[:gidx])
            avg_o = (sum_h + sum_a) / (2 * gidx)
        guess_o = np.ones(teams.shape[0]) * avg_o
        guess_d = np.ones(teams.shape[0]) * avg_o
        return guess_o, guess_d

    @staticmethod
    def _empty_iteration_summary(**kwargs):
        summary = {'o_residual': [], 'd_residual': [], 'game_index': 0,
                    'date': None, 'iterations': 0}
        for k, v in kwargs.iteritems():
            summary[k] = v
        return summary

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
    def _weights(n, n_pre=5, method='uniform'):
        """
        Initialize data vectors before iteration begins.

        Behavior:
            -TODO
            -Uses a preseason weight that dies out linearly with the number
             of games.

        Parameters
        ----------
        n : int
            The number of game weights to return.
        n_pre : int (default=5)
            The number of games before the preseason should no longer have an effect.

        Returns
        -------
        w : 1d numpy array
            A vector of game weights.
        w_pre : float
            The preseason rating weight.
        """
        n = int(n)
        if n == 0:
            return np.array([0.0]), 1.
        elif method == 'linear':
            w = np.arange(1, n + 1).astype(float)
        elif method == 'uniform':
            w = np.ones(n) / n
        elif method == 'log':
            w = np.log(np.arange(2, n + 2))
        elif method == 'exponential':
            w = np.exp(np.arange(1, n + 1) / 10.)
        else:
            raise ValueError("unsupported weighting method: %s" % method)
        w_pre = max(0, 1. - n / (n_pre + 1.))
        w_norm = 1 - w_pre
        w /= (w.sum() / w_norm)
        return w, w_pre

    @staticmethod
    def pythag(off_rtg, def_rtg, exponent):
        """
        Pythagorean expectation.

        Formula:
            \frac{off_{rtg}^{exponent}}{\left(off_{rtg}^{exponent} +
                def_{rtg}^{exponent}\right)}

        References
        ----------
        TODO

        Parameters
        ----------
        off_rtg : float or numpy array
            Offensive rating.
        def_rtg : float or numpy array
            Defensive rating.

        Returns
        -------
        pythag : float or numpy array
            The pythagorean expectation.
        """
        exp = exponent
        return off_rtg**(exp) / (off_rtg**(exp) + def_rtg**(exp))

    @staticmethod
    def log5(pa, pb):
        """Log5 win probability."""
        return (pa - pa * pb) / (pa + pb - 2 * pa * pb)

    @staticmethod
    def home_win_prob(home_o, home_d, away_o, away_d, neutral):
        """Get the home win probabilities for games, given home and away ratings."""
        home_adjustments = np.array(map(lambda x: 1.0 if x else AdjustedStat.home_factor, neutral))
        away_adjustments = 1.0 / home_adjustments
        home_o *= home_adjustments
        home_d /= home_adjustments
        away_o *= away_adjustments
        away_d /= away_adjustments
        phome = AdjustedStat.pythag(home_o, home_d, AdjustedStat.pythag_exponent)
        paway = AdjustedStat.pythag(away_o, away_d, AdjustedStat.pythag_exponent)
        return AdjustedStat.log5(phome, paway)

    @staticmethod
    def _upset_weights(upset_factors):
        alpha, beta, loc, scale = .6178732886, 7317446.69798, -0.17565759957, 163983.937878
        return 1 - scs.beta.cdf(upset_factors, alpha, beta, loc, scale)

    @staticmethod
    def _weight_by_prediction(home_o, home_d, away_o, away_d, neutral, home_scores, away_scores):
        """Weights that emphasize close games more than blowouts."""
        home_probabilities = AdjustedStat.home_win_prob(home_o, home_d, away_o, away_d, neutral)
        diff_ratio = (home_scores - away_scores) / (home_scores + away_scores)
        upset_factors = home_probabilities * diff_ratio
        weights = AdjustedStat._upset_weights(upset_factors)
        return np.ones(neutral.shape[0]).astype(float)

    @staticmethod
    def _combine_weights(w1, w2, sum_to=1.):
        w = w1 * w2
        return (w / np.sum(w).astype(float)) * sum_to

    @staticmethod
    def _loc_factor(is_home, is_neutral):
        """The location multiplier for a game."""
        if is_neutral:
            loc_factor = 1.
        elif is_home:
            loc_factor = 1. / AdjustedStat.home_factor
        else:
            loc_factor = AdjustedStat.home_factor
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
        return np.linalg.norm(old - new) / np.linalg.norm(old)

    @staticmethod
    def _check_convergence(residual, tol):
        """Check if a stat vector has converged."""
        assert residual >= 0.0, 'residual should be positive but was %s' % residual
        return residual < tol

    @staticmethod
    def _is_converged(residual_o, residual_d, tol=0.0005):
        """Check if the rate algorithm has converged."""
        return AdjustedStat._check_convergence(residual_o, tol) and \
            AdjustedStat._check_convergence(residual_d, tol)

    def _should_rate(self, iter, num_games):
        # compute every game_skip games and on the final iteration
        return iter % self.game_skip == 0 or iter == num_games - 1

    def _rate_multiple2(self, unstacked):
        """
        Get adjusted stat ratings across multiple seasons. This method should
        be called when the `rate` method detects that it was given data from
        multiple seasons.

        Parameters
        ----------
        unstacked : dataframe
            Unstacked dataframe containing game and stat information.

        Returns
        -------
        models : list
            A list of `AdjustedStat` models.
        unstacked : dataframe
            A single unstacked dataframe containing ratings for all seasons.
        """
        unstacked['season'] = unstacked['dt'].map(org_ncaa.get_season)
        seasons = RatingsModel._get_seasons(unstacked)
        dfs = []
        models = []
        for season in seasons:
            print 'Rating for season: %s' % season
            u = unstacked[unstacked['season'] == season]
            adj = AdjustedStat(self.stat)
            u = adj.rate(u)
            dfs.append(u)
            models.append(adj)

        return models, pd.concat(dfs)

    def _rate_multiple(self, unstacked):
        """
        Get adjusted stat ratings across multiple seasons. This method should
        be called when the `rate` method detects that it was given data from
        multiple seasons.

        Parameters
        ----------
        unstacked : dataframe
            Unstacked dataframe containing game and stat information.

        Returns
        -------
        models : list
            A list of `AdjustedStat` models.
        unstacked : dataframe
            A single unstacked dataframe containing ratings for all seasons.
        """
        unstacked['season'] = unstacked['dt'].map(org_ncaa.get_season)
        seasons = RatingsModel._get_seasons(unstacked)
        dfs = []
        for season in seasons:
            print 'Rating for season: %s' % season
            u = unstacked[unstacked['season'] == season]
            u = self.rate(u)
            dfs.append(u)

        return pd.concat(dfs)

    @staticmethod
    def _rate_for_games(unstacked, time_vector, omatrix, dmatrix, stat_name=""):
        """
        From the incremental adjusted ratings, compute vectors of home
        offensive and defensive and away offensive and defensive ratings.

        Parameters
        ----------
        unstacked : dataframe
            Unstacked dataframe containing game and stat information.
        time_vector : array_like
            Vector of time indices.
        omatrix : array_like
            Array of offensive ratings vectors of length `len(time_vector)`.
        dmatrix : array_like
            Array of defensive ratings vectors of length `len(time_vector)`.

        Returns
        -------
        unstacked : dataframe
            Unstacked dataframe with computed adjusted stats added to it.
        """
        h_off = np.zeros(unstacked.shape[0])
        h_def = np.zeros(unstacked.shape[0])
        a_off = np.zeros(unstacked.shape[0])
        a_def = np.zeros(unstacked.shape[0])
        for j, (idx1, idx2) in enumerate(zip(time_vector, np.roll(time_vector, -1))[:-1]):
            hindices = unstacked.i_hteam.values[idx1:idx2]
            aindices = unstacked.i_ateam.values[idx1:idx2]
            h_off_ratings = omatrix[j][hindices]
            h_def_ratings = dmatrix[j][hindices]
            a_off_ratings = omatrix[j][aindices]
            a_def_ratings = dmatrix[j][aindices]
            h_off[idx1:idx2] = h_off_ratings
            h_def[idx1:idx2] = h_def_ratings
            a_off[idx1:idx2] = a_off_ratings
            a_def[idx1:idx2] = a_def_ratings
        col_prefixes = ['h_adj_', 'h_adj_d', 'a_adj_', 'a_adj_d']
        col_names = map(lambda col: col + stat_name, col_prefixes)
        data = zip(col_names, [h_off, h_def, a_off, a_def])
        for name, col in data:
            unstacked[name] = col
        return unstacked

    def _rate_one(self, oraw, draw, avg_o, avg_d, loc, idx,
                  current_index, o_pre, d_pre, start_o, start_d):
        """
        Run an adjusted stat model rating for the games data provided.

        Stats are adjusted according to:
            adj = \sum raw_stat / adj_opp_stat * avg_stat * w_i * \\
                  loc_i + w_pre * stat_pre

        References
        ----------
        -Kenpom's own ratings explanation
            http://kenpom.com/blog/index.php/weblog/entry/ratings_explanation
        -Kenpom's explanation of margin of victory adjustment:
            http://kenpom.com/blog/index.php/weblog/entry/pomeroy_ratings_version_2.0
        -Kenpom's adjusted stats calculations explanation:
            http://kenpom.com/blog/index.php/weblog/entry/national_efficiency/

        Parameters
        ----------
        oraw : list
            List of raw offensive outputs for each team for each game.
        draw : list
            List of raw defensive outputs for each team for each game.
        avg_o : float
            Average offensive output for all games.
        avg_d : float
            Average defensive output for all games.
        loc : list
            List of location factors for each team for each game
        idx : list
            List of game indices for each team for each game.
        current_index : dict
            Dictionary of how many games to include for each team
        o_pre : 1d numpy array
            Preseason offensive ratings.
        d_pre : 1d numpy array
            Preseason defensive ratings
        start_o : 1d numpy array
            Initial guess at offensive ratings before iteration.
        start_d : 1d numpy array
            Initial guess at defensive ratings before iteration.


        Returns
        -------
        unstacked : dataframe
            Original unstacked dataframe with ratings columns appended.
        """
        def _empty_iteration_summary(**kwargs):
            summary = {'o_residual': [], 'd_residual': [], 'game_index': 0,
                        'date': None, 'iterations': 0}
            for k, v in kwargs.iteritems():
                summary[k] = v
            return summary
        iteration_results = _empty_iteration_summary()

        adj_o = start_o
        adj_d = start_d

        num_teams = len(idx)

        if self.cache_intermediate:
            iteration_results['o_history'] = [adj_o.copy()]
            iteration_results['d_history'] = [adj_d.copy()]

        for i in xrange(self.num_iterations):
            old_adj_o = adj_o.copy()
            old_adj_d = adj_d.copy()
            for j in xrange(num_teams):
                # the number of games to include for the team
                k = current_index[j]
                if k == 0:
                    continue

                w_time, w_pre = AdjustedStat._weights(k, self.n_pre, method='exponential')
                w = w_time
                adj_o[j] = AdjustedStat._adjust(oraw[j][:k], adj_d[idx[j][:k]],
                                                avg_o, w, loc[j][:k], w_pre, o_pre[j])
                adj_d[j] = AdjustedStat._adjust(draw[j][:k], adj_o[idx[j][:k]],
                                                avg_d, w, 1. / loc[j][:k], w_pre, d_pre[j])
            if self.cache_intermediate:
                iteration_results['o_history'].append(adj_o.copy())
                iteration_results['d_history'].append(adj_d.copy())
            oresidual = AdjustedStat._calc_residual(old_adj_o, adj_o)
            dresidual = AdjustedStat._calc_residual(old_adj_d, adj_d)
            iteration_results['o_residual'].append(oresidual)
            iteration_results['d_residual'].append(dresidual)

            if AdjustedStat._is_converged(oresidual, dresidual):
                pass

        iteration_results['iterations'] = i + 1

        return adj_o, adj_d, iteration_results

    def rate(self, unstacked):
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
        -Kenpom's own ratings explanation
            http://kenpom.com/blog/index.php/weblog/entry/ratings_explanation
        -Kenpom's explanation of margin of victory adjustment:
            http://kenpom.com/blog/index.php/weblog/entry/pomeroy_ratings_version_2.0
        -Kenpom's adjusted stats calculations explanation:
            http://kenpom.com/blog/index.php/weblog/entry/national_efficiency/

        Parameters
        ----------
        unstacked : dataframe
            Unstacked dataframe containing game and stat information.

        Returns
        -------
        unstacked : dataframe
            Original unstacked dataframe with ratings columns appended.
        """
        util.validate_games(unstacked, ['pts', 'poss', 'ppp'])
        if AdjustedStat._is_multiple_seasons(unstacked):
            return self._rate_multiple(unstacked)

        # need games to be in sequential order
        unstacked = unstacked.sort('dt')
        teams, team_index = AdjustedStat._get_teams(unstacked)
        num_teams = teams.shape[0]
        num_games = unstacked.shape[0]
        unstacked = AdjustedStat._add_team_index(unstacked, team_index)

        idx, loc, oraw, draw = self._initialize(unstacked, teams)
        o_pre, d_pre = self._preseason_rank(teams)

        # Add the preseason rank as a starting point
        adj_o_history = [o_pre]
        adj_d_history = [d_pre]

        game_indices = unstacked[['i_hteam', 'i_ateam']].values
        current_index = {team: 0 for team in xrange(num_teams)}
        dates = unstacked['dt'].values
        games_included = [0]
        zero_summary = AdjustedStat._empty_iteration_summary(date=dates[0])

        results = [zero_summary]
        for gidx, (hidx, aidx) in enumerate(game_indices):
            # increment team vector indices to include new game
            current_index[hidx] += 1
            current_index[aidx] += 1
            if not self._should_rate(gidx, num_games):
                continue

            if self.verbose:
                print 'No. of games included: %s' % gidx

            avg_o, avg_d = self._average_stats(oraw, draw, current_index)
            if gidx == 0:
                adj_o, adj_d = self._initial_guess(unstacked, teams, gidx)
            else:
                # the initial guess is simply the ratings from the previous iteration
                # TODO: some weird convergence issues for this method
                adj_o = adj_o.copy()
                adj_d = adj_d.copy()

            adj_o, adj_d, iter_results = \
                self._rate_one(oraw, draw, avg_o, avg_d, loc, idx, current_index,
                               o_pre, d_pre, start_o=adj_o, start_d=adj_d)
            adj_o_history.append(adj_o.copy())
            adj_d_history.append(adj_d.copy())
            results.append(iter_results)
            games_included.append(gidx + 1)

        # add a rating column to include ratings for each game in the dataframe
        unstacked = AdjustedStat._rate_for_games(unstacked, games_included, adj_o_history, adj_d_history, self.stat)
        self.offensive_ratings = np.array(adj_o_history)
        self.defensive_ratings = np.array(adj_d_history)
        self.results = results
        self.team_index = team_index
        return unstacked

    def rate2(self, unstacked):
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
        -Kenpom's own ratings explanation
            http://kenpom.com/blog/index.php/weblog/entry/ratings_explanation
        -Kenpom's explanation of margin of victory adjustment:
            http://kenpom.com/blog/index.php/weblog/entry/pomeroy_ratings_version_2.0
        -Kenpom's adjusted stats calculations explanation:
            http://kenpom.com/blog/index.php/weblog/entry/national_efficiency/

        Parameters
        ----------
        unstacked : dataframe
            Unstacked dataframe containing game and stat information.

        Returns
        -------
        unstacked : dataframe
            Original unstacked dataframe with ratings columns appended.
        """
        util.validate_games(unstacked, ['pts', 'poss', 'ppp'])
        if AdjustedStat._is_multiple_seasons(unstacked):
            return self._rate_multiple(unstacked)

        # need games to be in sequential order
        unstacked = unstacked.sort('dt')
        teams, team_index = AdjustedStat._get_teams(unstacked)
        # self.add_team_index(unstacked) ?
        unstacked['i_hteam'] = unstacked['hteam_id'].map(lambda tid: team_index[tid])
        unstacked['i_ateam'] = unstacked['ateam_id'].map(lambda tid: team_index[tid])

        def _empty_iteration_summary(**kwargs):
            summary = {'o_residual': [], 'd_residual': [], 'game_index': 0,
                        'date': None, 'iterations': 0}
            for k, v in kwargs.iteritems():
                summary[k] = v
            return summary

        idx, loc, oraw, draw = self._initialize(unstacked, teams)
        o_pre, d_pre = self._preseason_rank(teams)

        num_teams = len(idx)

        # Add the preseason rank as a starting point
        adj_o_history = [o_pre]
        adj_d_history = [d_pre]

        game_indices = unstacked[['i_hteam', 'i_ateam']].values
        current_index = {team: 0 for team in xrange(num_teams)}
        dates = unstacked['dt'].values
        time = [0]
        zero_summary = _empty_iteration_summary(date=dates[0])
        # keep track of adjusted ratings for each game as we iterate
        home_o = o_pre[game_indices[:, 0]]
        home_d = d_pre[game_indices[:, 0]]
        away_o = o_pre[game_indices[:, 1]]
        away_d = d_pre[game_indices[:, 1]]
        prev_idx = 0
        results = [zero_summary]
        for gidx, (hidx, aidx) in enumerate(game_indices):
            # increment team vector indices to include new game
            current_index[hidx] += 1
            current_index[aidx] += 1
            if gidx % self.game_skip != 0 and gidx != game_indices.shape[0] - 1:
                continue

            if self.verbose:
                print 'No. of games included: %s' % gidx

            avg_o, avg_d = self._average_stats(oraw, draw, current_index)
            if gidx == 0:
                adj_o, adj_d = self._initial_guess(unstacked, teams, gidx)
            else:
                # the initial guess is simply the ratings from the previous iteration
                # TODO: some weird convergence issues for this method
                adj_o = adj_o.copy()
                adj_d = adj_d.copy()

            iteration_results = _empty_iteration_summary()

            if self.cache_intermediate:
                iteration_results['o_history'] = [adj_o.copy()]
                iteration_results['d_history'] = [adj_d.copy()]

            for i in xrange(self.num_iterations):
                old_adj_o = adj_o.copy()
                old_adj_d = adj_d.copy()
                for j in xrange(num_teams):
                    # the number of games to include for the team
                    k = current_index[j]
                    if k == 0:
                        continue

                    w_time, w_pre = AdjustedStat._weights(k, self.n_pre, method='exponential')
                    w_mov = AdjustedStat._weight_by_prediction(home_o[idx[j][:k]], home_d[idx[j][:k]],
                                                                 away_o[idx[j][:k]], away_d[idx[j][:k]],
                                                                 unstacked.neutral.values[idx[j][:k]],
                                                                 unstacked.hpts.values[idx[j][:k]],
                                                                 unstacked.apts.values[idx[j][:k]])
                    w = AdjustedStat._combine_weights(w_time, w_mov, sum_to=(1 - w_pre))
                    w = w_time
                    adj_o[j] = AdjustedStat._adjust(oraw[j][:k], adj_d[idx[j][:k]],
                                                    avg_o, w, loc[j][:k], w_pre, o_pre[j])
                    adj_d[j] = AdjustedStat._adjust(draw[j][:k], adj_o[idx[j][:k]],
                                                    avg_d, w, 1. / loc[j][:k], w_pre, d_pre[j])
                if self.cache_intermediate:
                    iteration_results['o_history'].append(adj_o.copy())
                    iteration_results['d_history'].append(adj_d.copy())
                oresidual = AdjustedStat._calc_residual(old_adj_o, adj_o)
                dresidual = AdjustedStat._calc_residual(old_adj_d, adj_d)
                iteration_results['o_residual'].append(oresidual)
                iteration_results['d_residual'].append(dresidual)

                if AdjustedStat._is_converged(oresidual, dresidual):
                    pass

            iteration_results['iterations'] = i + 1
            time.append(gidx + 1)
            iteration_results['game_index'] = gidx + 1
            iteration_results['date'] = dates[gidx]
            adj_o_history.append(adj_o.copy())
            adj_d_history.append(adj_d.copy())

            home_indices = unstacked['i_hteam'].values[prev_idx:gidx]
            away_indices = unstacked['i_ateam'].values[prev_idx:gidx]
            home_o[prev_idx:gidx] = adj_o[home_indices]
            home_d[prev_idx:gidx] = adj_d[home_indices]
            away_o[prev_idx:gidx] = adj_o[away_indices]
            away_d[prev_idx:gidx] = adj_d[away_indices]

            results.append(iteration_results)

        # add a rating column to include ratings for each game in the dataframe
        unstacked = AdjustedStat._rate_for_games(unstacked, time, adj_o_history, adj_d_history, self.stat)
        self.offensive_ratings = np.array(adj_o_history)
        self.defensive_ratings = np.array(adj_d_history)
        self.results = results
        self.team_index = team_index
        return unstacked

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

    def _assert_trained(self):
        assert getattr(self, 'results') is not None, \
            "the model has not been trained!"

    def ratings_df(self):
        """Get dataframes of the offensive and defensive ratings over time."""
        self._assert_trained()
        indices = np.array([res['game_index'] for res in self.results])
        dates = np.array([res['date'] for res in self.results])
        reverse_index = {v: k for k, v in self.team_index.iteritems()}
        columns = [reverse_index[i] for i in xrange(len(reverse_index))]
        df_off = pd.DataFrame(self.offensive_ratings, columns=columns)
        df_def = pd.DataFrame(self.defensive_ratings, columns=columns)
        df_off['game_index'] = indices
        df_off['date'] = dates
        df_def['game_index'] = indices
        df_def['date'] = dates
        return df_off.set_index(['date', 'game_index']), df_def.set_index(['date', 'game_index'])

    def __repr__(self):
        param_string = ['%s=%s' % (k, v) for k, v in self.params.iteritems()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(param_string))

if __name__ == "__main__":
    unstacked1, stacked1, teams1 = util.get_data(2013)
    unstacked2, stacked2, teams2 = util.get_data(2016)
    unstacked = pd.concat([unstacked1, unstacked2])
    adj = AdjustedStat('ppp', n_pre=5, num_iterations=2, game_skip=30)
    models, u = adj.rate(unstacked)
