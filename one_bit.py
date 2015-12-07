import numpy as np

class OneBitAggregator(object):

    def __init__(self):
        pass

    def rate_for_games(self, teams, games):
        return {'one_bit': np.ones(games.shape[0])}