from abc import abstractmethod

import numpy as np

import org_ncaa

class RatingsModel(object):

    def __init__(self):
        pass

    @abstractmethod
    def _default_params(self):
        raise NotImplementedError

    def set_params(self, **kwargs):
        for k, v in kwargs.iteritems():
            if k in self._default_params():
                setattr(self, k, v)
                self.params[k] = v
            else:
                print 'AdjustedStat class does not accept %s as a ' \
                      'parameter and will be ignored' % k

    def get_param(self, param):
        return getattr(self, param)

    @abstractmethod
    def rate(self, unstacked):
        raise NotImplementedError

    @staticmethod
    def _get_teams(unstacked, min_games=0):
        """Get teams and index from the unstacked games dataframe."""
        count_col = 'game_id'
        home_gb = unstacked.groupby('hteam_id').count()[[count_col]]
        away_gb = unstacked.groupby('ateam_id').count()[[count_col]]
        merged = home_gb.merge(away_gb, how='outer', left_index=True, right_index=True)
        merged[count_col] = merged[count_col + '_x'] + merged[count_col + '_y']
        teams = merged.reset_index().rename(columns={'index': 'team_id'})
        teams = teams.sort('team_id')
        teams['i_team'] = np.arange(teams.shape[0])
        return teams[['team_id', 'i_team']]

    @staticmethod
    def _get_seasons(unstacked):
        return np.unique(unstacked['dt'].map(org_ncaa.get_season))

    @staticmethod
    def _is_multiple_seasons(unstacked):
        seasons = RatingsModel._get_seasons(unstacked)
        return seasons.shape[0] > 1

    @abstractmethod
    def _rate_multiple(self, unstacked):
        raise NotImplementedError




