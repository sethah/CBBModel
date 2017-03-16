import pandas as pd
import numpy as np

from models.RatingsModel import RatingsModel
from models.RatingsHistory import RatingsHistory
from algos.RLS import RLS

class Massey(RatingsModel):

    def __init__(self):
        self.ratings_history = None

    def _default_params(self):
        return {}

    @property
    def hashistory(self):
        return self.ratings_history is not None

    @staticmethod
    def _get_home(neutral, teamid, homeid):
        if neutral:
            return 0
        elif teamid == homeid:
            return 1
        else:
            return -1

    @staticmethod
    def _constrained_dummies(dummies):
        N, k = dummies.shape
        subtract = np.array([np.zeros(k - 1) if dummies.values[i, -1] == 0 else np.ones(k - 1) for i in range(N)])
        return pd.DataFrame(dummies.values[:, :-1] - subtract, columns=dummies.columns[:-1])

    @staticmethod
    def _construct_data(stacked):
        stacked = stacked.sort('dt').reset_index().drop('index', 1)
        stacked['home'] = stacked.apply(lambda row: Massey._get_home(row.neutral, row.team_id, row.hteam_id), 1)
        feature_df = stacked[['game_id', 'dt', 'i_team', 'i_opp', 'home', 'ppp']]
        off_dummies = pd.get_dummies(feature_df['i_team'].astype(int), prefix='off')
        def_dummies = pd.get_dummies(feature_df['i_opp'].astype(int), prefix='def')
        glm_df = pd.concat([feature_df, Massey._constrained_dummies(off_dummies),
                            Massey._constrained_dummies(def_dummies)], 1)
        glm_df['const'] = 1
        return glm_df

    @staticmethod
    def _get_data_matrices(glm_df, num_teams):
        feature_columns = ['off_%s' % i for i in range(num_teams - 1)] + \
                          ['def_%s' % i for i in range(num_teams - 1)] + \
                          ['home'] + ['const']
        X = glm_df[feature_columns].values
        y = glm_df['ppp'].values
        return X, y

    def _append_ratings(self, unstacked):
        n = unstacked.shape[0]
        hortg, hdrtg, aortg, adrtg = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        for i, row in unstacked.iterrows():
            hortg[i], hdrtg[i] = self.get_rating_at_date(row['i_hteam'], row['dt'])
            aortg[i], adrtg[i] = self.get_rating_at_date(row['i_ateam'], row['dt'])

        unstacked['hortg'] = hortg
        unstacked['hdrtg'] = hdrtg
        unstacked['aortg'] = aortg
        unstacked['adrtg'] = adrtg
        return unstacked

    def rate(self, unstacked):
        pass

    def _rate_every_game(self, stacked, teams):
        df = Massey._construct_data(stacked)
        self.num_teams = teams.shape[0]
        X, y = Massey._get_data_matrices(df, self.num_teams)
        algo = RLS()
        algo.fit(X, y, keep_history=True)
        self.ratings_history = MasseyHistory(algo.history, df['dt'].values, df['game_id'].values, teams.shape[0])

    def get_rating_at_date(self, idx, dt):
        return self.ratings_history.get_rating_at_date(dt, idx)


class MasseyHistory(RatingsHistory):

    def _get_coef(self, history_idx, team_idx, offset=0):
        assert team_idx < self.num_teams
        if team_idx == offset + self.num_teams:
            return -1 * np.sum(self.ratings[history_idx][offset:offset + self.num_teams - 1])
        else:
            return self.ratings[history_idx][int(offset + team_idx)]

    def _get_off_coef(self, history_idx, team_idx):
        return self._get_coef(history_idx, team_idx)

    def _get_def_coef(self, history_idx, team_idx):
        return self._get_coef(history_idx, team_idx, self.num_teams - 1)

    # TODO: some of these should go in parent class
    def _get_coefs_at_date(self, dt):
        # TODO: if date isn't found then get the closest one
        dt_idx = self.date_counts[dt] - 1
        return self.history['ratings'][dt_idx]

    def _get_idx_from_dt(self, dt):
        return self.date_counts[dt] - 1

    def get_rating_at_date(self, dt, team_idx):
        dt_idx = self._get_idx_from_dt(dt)
        return self._get_off_coef(dt_idx, team_idx), self._get_def_coef(dt_idx, team_idx)

    def construct_df(self):
        """
        Method that makes the history dataframe.

        We should add the missing "pivot" team into this dataframe. Then make it easy
        to index by the date and the team. Add the game_id and the dt as a datetime_index.
        """
        feature_columns = ['off_%s' % i for i in range(self.num_teams - 1)] + \
                          ['def_%s' % i for i in range(self.num_teams - 1)] + \
                          ['home'] + ['intercept']
