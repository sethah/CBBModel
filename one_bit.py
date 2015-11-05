import numpy as np

def rate(games, rating_col='rating'):
    games[rating_col] = np.ones(games.shape[0])
    return games