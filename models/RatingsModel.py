from abc import abstractmethod

import numpy as np
import pandas as pd

import org_ncaa

class RatingsModel(object):

    def __init__(self):
        pass

    @abstractmethod
    def _default_params(self):
        raise NotImplementedError

    @abstractmethod
    def rate(self, unstacked):
        raise NotImplementedError

    # @abstractmethod
    # def _rate_multiple(self, unstacked):
    #     raise NotImplementedError

    def set_params(self, **kwargs):
        """Set parameters in the dictionary."""
        for param_name, value in kwargs.iteritems():
            # only set parameters that are in the default
            if param_name in self._default_params():
                setattr(self, param_name, value)
                self.params[param_name] = value
            else:
                print('AdjustedStat class does not accept %s as a ' \
                      'parameter and will be ignored' % param_name)

    def get_param(self, param_name):
        return getattr(self, param_name)

    @staticmethod
    def _add_team_index(unstacked, team_index):
        unstacked['i_hteam'] = unstacked['hteam_id'].map(lambda tid: team_index[tid])
        unstacked['i_ateam'] = unstacked['ateam_id'].map(lambda tid: team_index[tid])
        return unstacked

    @staticmethod
    def _get_teams(unstacked, min_games=0):
        """Get teams and index from the unstacked games dataframe."""
        count_col = 'game_id'
        home_gb = unstacked.groupby('hteam_id').count()[[count_col]]
        away_gb = unstacked.groupby('ateam_id').count()[[count_col]]
        merged = home_gb.merge(away_gb, how='outer', left_index=True, right_index=True)
        merged[count_col] = merged[count_col + '_x'] + merged[count_col + '_y']
        merged.index.name = 'index'
        teams = merged.reset_index().rename(columns={'index': 'team_id'})
        teams = teams.sort('team_id')
        teams['i_team'] = np.arange(teams.shape[0])
        team_index = {}
        for team, idx in teams[['team_id', 'i_team']].values:
            team_index[int(team)] = int(idx)
        return teams[['team_id', 'i_team']], team_index

    @staticmethod
    def _get_seasons(unstacked):
        return np.unique(unstacked['dt'].map(org_ncaa.get_season))

    @staticmethod
    def _is_multiple_seasons(unstacked):
        seasons = RatingsModel._get_seasons(unstacked)
        return seasons.shape[0] > 1

    def _rate_multiple(self, unstacked):
        unstacked['season'] = unstacked['dt'].map(org_ncaa.get_season)
        seasons = RatingsModel._get_seasons(unstacked)
        dfs = []
        for season in seasons:
            print('Rating for season: %s' % season)
            u = unstacked[unstacked['season'] == season]
            u = self.rate(u)
            dfs.append(u)

        return pd.concat(dfs)
