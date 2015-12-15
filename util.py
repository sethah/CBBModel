import pandas as pd
import numpy as np
import datetime

from DB import DB

ALL_SEASONS = [2010, 2011, 2012, 2013, 2014, 2015]

def get_season(dt):
    """Assign a season to a given date."""
    return dt.year if dt.month <= 6 else dt.year + 1

def safe_divide(num, den):
    return np.nan_to_num(num / den)

def get_season_bounds(input):

    def _get_season_years(season):
        return (season - 1, season)

    if type(input) == int:
        years = _get_season_years(input)
    elif type(input) == str:
        dt = datetime.strptime(input, '%m/%d/%Y')
        years = _get_season_years(get_season(dt))
    elif type(input) == datetime.date:
        years = _get_season_years(get_season(input))

    return (datetime.date(years[0], 10, 30), datetime.date(years[1], 4, 30))


def get_games(input=None):
    """
    Get games from database and sort by date ascending
    :param season: INT - season to filter games by
    :return: DATAFRAME
    """
    if type(input) == str:
        # try to convert to date
        dt = datetime.strptime(input, '%m/%d/%Y').date()
        bounds = map(lambda d: d.strftime('%Y-%m-%d'), list(get_season_bounds(dt)))
        bounds = [bounds[0], input]
    elif isinstance(input, datetime.date):
        bounds = map(lambda d: d.strftime('%Y-%m-%d'), list(get_season_bounds(input)))
        bounds = [bounds[0], input.strftime('%Y-%m-%d')]
    elif type(input) == int and (input > 1900 and input <= get_season(datetime.datetime.now())):
        bounds = map(lambda d: d.strftime('%Y-%m-%d'), list(get_season_bounds(input)))
    else:
        bounds = [datetime.date(1900, 10, 30), datetime.date(3000, 4, 30)]
    games = pd.read_sql("SELECT * FROM games_test WHERE dt > '%s' AND dt <= '%s'" % tuple(bounds), DB.conn)
    return games

def get_teams(games):
    """
    Create a dataframe of all teams for a set of games.
    :param games: DATAFRAME - games data
    :return: DATAFRAME - unique set of teams
    """
    hteams = games[['hteam_id']]
    ateams = games[['ateam_id']]
    hteams = hteams.rename(columns={'hteam_id': 'team_id'})
    ateams = ateams.rename(columns={'ateam_id': 'team_id'})
    teams = pd.concat([hteams, ateams], axis=0)
    teams.drop_duplicates(inplace=True)
    teams.dropna(inplace=True)
    return teams