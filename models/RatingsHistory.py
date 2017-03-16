from abc import abstractmethod
from collections import Counter

import numpy as np
import pandas as pd

class RatingsHistory(object):

    def __init__(self, ratings, dates, game_ids, num_teams):
        self.num_teams = num_teams
        self.ratings = ratings
        self.dates = dates
        self.game_ids = game_ids
        self._date_counts = None

    @property
    def date_counts(self):
        if self._date_counts is None:
            c = Counter(self.dates)
            dt_counts = []
            for dt, cnt in zip(c.keys(), c.values()):
                dt_counts.append([dt, cnt])
            dt_counts = np.array(sorted(dt_counts, key=lambda x: x[0]))
            dt_counts = pd.Series(np.cumsum(dt_counts[:, 1]), dt_counts[:, 0])
            self._date_counts = dt_counts
        return self._date_counts

    @abstractmethod
    def _get_off_coef(self, history_idx, team_idx):
        raise NotImplementedError

    @abstractmethod
    def _get_def_coef(self, history_idx, team_idx):
        raise NotImplementedError
