import numpy as np
import pandas as pd

import util

import org_ncaa
import statsmodels.api as sm

class Massey(object):

    def __init__(self):
        self.ratings = ['home_off', 'home_def', 'away_off', 'away_def']
        self.feature_columns = None
        self.num_teams = None

    def _default_params(self):
        return {}

    def rate(self, unstacked):
        pass

    def get_feature_index(self, feature_name):
        if feature_name in self.feature_columns:
            return self.feature_columns.index(feature_name)
        else:
            return None

    @staticmethod
    def _get_feature_cols(df):
        off_cols = list(filter(lambda c: 'off' in c, df.columns.values))
        def_cols = list(filter(lambda c: 'def' in c, df.columns.values))
        return off_cols + def_cols + ['home', 'const']

    def rate_one(self, stacked, unstacked):
        data = Massey._get_data_matrix(stacked)
        self.num_teams = len(list(filter(lambda c: 'off' in c, data.columns.values))) + 1
        self.feature_columns = Massey._get_feature_cols(data)
        X = data[self.feature_columns].values
        y = data['ppp'].values
        return self._rate_one(X, y)

    def _rate_one(self, X, y):
        res = sm.GLM(y, X).fit()
        params = Massey.extract_params(res, self.num_teams, self.feature_columns)
        return params

    def _rate_for_season(self, stacked, unstacked, prior=None):
        data = Massey._get_data_matrix(stacked)
        self.num_teams = len(list(filter(lambda c: 'off' in c, data.columns.values))) + 1
        self.feature_columns = Massey._get_feature_cols(data)
        X = data[self.feature_columns].values
        y = data['ppp'].values
        if prior:
            X, y, prior_length = self._add_prior(X, y, prior, self.num_teams)
        else:
            prior_length = 0
        indices = Massey.get_indices(stacked, 2000, 600)
        prev_idx = indices[0]
        results = []
        start_params = None
        for i, idx in enumerate(indices):
            res = sm.GLM(y[:prior_length + prev_idx], X[:prior_length + prev_idx]).fit(start_params=start_params)
            # params = Massey.extract_params(res, self.num_teams, self.feature_columns)
            start_params = res.params
            lb, ub = prev_idx / 2, idx / 2
            results.append((lb, ub, res))
            print("Iteration % s" % i)
            # params = self._rate_one(X, y)

        home_off_ratings = np.ones(unstacked.shape[0]) * -1
        home_def_ratings = np.ones(unstacked.shape[0]) * -1
        away_off_ratings = np.ones(unstacked.shape[0]) * -1
        away_def_ratings = np.ones(unstacked.shape[0]) * -1
        home_factors = np.ones(unstacked.shape[0]) * -1
        home_indices = unstacked['i_hteam'].values
        away_indices = unstacked['i_ateam'].values
        for lb, ub, res in results:
            params = Massey.extract_params(res, self.num_teams, self.feature_columns)
            home_idx = home_indices[lb:ub]
            away_idx = away_indices[lb:ub]
            home_off_ratings[lb:ub] = params['off'][home_idx]
            home_def_ratings[lb:ub] = params['def'][home_idx]
            away_off_ratings[lb:ub] = params['off'][away_idx]
            away_def_ratings[lb:ub] = params['def'][away_idx]
            home_factors[lb:ub] = params['home']
        return {'home_off': home_off_ratings, 'home_def': home_def_ratings, 'away_off': away_off_ratings,
                'away_def': away_def_ratings, 'home_factors': home_factors}

    @staticmethod
    def get_dates(stacked, approx_burn_games, approx_interval):
        date_counts = np.cumsum(stacked.groupby('dt').count()['game_id'])
        _next = approx_burn_games * 2
        dates = []
        for i, (dt, count) in enumerate(date_counts.iteritems()):
            if count > _next:
                dates.append(dt.date())
                _next += approx_interval * 2
        return dates

        # cum_indices = np.cumsum(date_counts)
        # next_ = cum_indices[-1]
        # indices = []
        # for gp in cum_indices[::-1]:
        #     if gp <= approx_burn_games:
        #         break
        #     if gp <= next_:
        #         indices.append(gp)
        #         next_ -= approx_interval
        # return indices[::-1]

    def _add_prior(self, X, y, priors, num_teams):
        X_off = Massey._get_prior_matrix_team('off', num_teams, X.shape[1])
        X_def = Massey._get_prior_matrix_team('def', num_teams, X.shape[1])
        X_home = Massey._get_prior_matrix_const(X.shape[1], self.get_feature_index("home"))
        X_intercept = Massey._get_prior_matrix_const(X.shape[1], self.get_feature_index("intercept"))
        Xnew = np.concatenate([X_off, X_def, X_home, X_intercept, X])
        ynew = np.concatenate([priors['off'], priors['def'],
                               np.array([priors['home'], priors['intercept']]), y])
        return Xnew, ynew, Xnew.shape[0] - X.shape[0]

    @staticmethod
    def _get_prior_matrix_const(num_cols, col_idx):
        prior_row = np.zeros(num_cols)
        prior_row[col_idx] = 1
        return prior_row[np.newaxis, :]

    @staticmethod
    def _get_prior_matrix_team(prior_type, num_teams, num_cols):
        start_col = 0 if prior_type == 'off' else num_teams - 1
        prior_rows = np.zeros((num_teams, num_cols))
        prior_rows[np.arange(num_teams - 1), np.arange(num_teams - 1) + start_col] = 1
        prior_rows[num_teams - 1, start_col:num_teams - 1 + start_col] = np.ones(num_teams - 1) * -1
        return prior_rows

    def _rate_multiple(self, unstacked):
        raise NotImplementedError

    @staticmethod
    def _get_data_matrix(stacked):
        stacked = stacked.sort('dt').reset_index().drop('index', 1)
        stacked['home'] = stacked.apply(lambda row: util.get_home(row.neutral, row.team_id, row.hteam_id), 1)
        feature_df = stacked[['dt', 'i_team', 'i_opp', 'home', 'ppp']]
        off_dummies = pd.get_dummies(stacked['i_team'].astype(int), prefix='off')
        def_dummies = pd.get_dummies(stacked['i_opp'].astype(int), prefix='def')
        ratings_df = pd.concat([feature_df, Massey._constrained_dummies(off_dummies),
                                Massey._constrained_dummies(def_dummies)], 1)
        ratings_df['const'] = 1
        return ratings_df

    @staticmethod
    def _constrained_dummies(dummies):
        N, k = dummies.shape
        subtract = np.array([np.zeros(k - 1) if dummies.values[i, -1] == 0 else np.ones(k - 1) for i in range(N)])
        return pd.DataFrame(dummies.values[:, :-1] - subtract, columns=dummies.columns[:-1])

    @staticmethod
    def extract_params(res, k, feature_cols):
        home_col = feature_cols.index("home") if "home" in feature_cols else None
        intercept_col = feature_cols.index("const") if "const" in feature_cols else None
        off_coefs = res.params[:k - 1]
        off_coefs = np.append(off_coefs, -1 * np.sum(off_coefs))
        def_coefs = res.params[k - 1:2 * (k - 1)]
        def_coefs = np.append(def_coefs, -1 * np.sum(def_coefs))

        coef_dict = {'off': off_coefs, 'def': def_coefs}
        if home_col:
            coef_dict['home'] = res.params[home_col]
        if intercept_col:
            coef_dict['intercept'] = res.params[intercept_col]
        return coef_dict
