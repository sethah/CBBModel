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

def get_date_bounds(input=None):
    if type(input) == str:
        # try to convert to date
        dt = datetime.datetime.strptime(input, '%m/%d/%Y').date()
        bounds = map(lambda d: d.strftime('%Y-%m-%d'), list(get_season_bounds(dt)))
        bounds = [bounds[0], dt.strftime('%Y-%m-%d')]
    elif isinstance(input, datetime.date):
        bounds = map(lambda d: d.strftime('%Y-%m-%d'), list(get_season_bounds(input)))
        bounds = [bounds[0], input.strftime('%Y-%m-%d')]
    elif type(input) == int and (input > 1900 and input <= get_season(datetime.datetime.now())):
        bounds = map(lambda d: d.strftime('%Y-%m-%d'), list(get_season_bounds(input)))
    else:
        bounds = [datetime.date(1900, 10, 30), datetime.date(3000, 4, 30)]

    return bounds

def get_games(input=None):
    """
    Get games from database and sort by date ascending
    :param season: INT - season to filter games by
    :return: DATAFRAME
    """
    bounds = get_date_bounds(input)
    games = pd.read_sql("SELECT * FROM games_test WHERE dt > '%s' AND dt <= '%s'" % tuple(bounds), DB.conn)
    return games

def get_teams(games):
    """
    Create a dataframe of all teams for a set of games.
    :param games: DATAFRAME - games data
    :return: DATAFRAME - unique set of teams
    """
    all_teams = pd.read_sql("SELECT ncaa, ncaaid FROM teams", DB.conn)
    hteams = games[['hteam_id']]
    ateams = games[['ateam_id']]
    hteams = hteams.rename(columns={'hteam_id': 'team_id'})
    ateams = ateams.rename(columns={'ateam_id': 'team_id'})
    teams = pd.concat([hteams, ateams], axis=0)
    teams.drop_duplicates(inplace=True)
    teams = teams.reset_index()
    teams['iteam'] = teams.index.values
    teams.drop('index', 1, inplace=True)
    teams.dropna(inplace=True)
    tmp = teams.merge(all_teams, left_on='team_id', right_on='ncaaid')
    return tmp[['iteam', 'team_id', 'ncaa']]

def get_data(input=None):
    bounds = get_date_bounds(input)
    q = """SELECT
                g.*,
                b.team,
                b.pts,
                b.fga - COALESCE(b.oreb, 0) + COALESCE(b.turnover, 0) + 0.475 * COALESCE(b.fta, 0) AS poss
            FROM box_test b
            JOIN games_test g
            ON b.game_id = g.game_id
            AND b.first_name='Totals'
            AND g.dt > '%s'
            AND g.dt <= '%s'
        """ % tuple(bounds)
    stacked = pd.read_sql(q, DB.conn).dropna(subset=['ateam_id', 'hteam_id'])
    all_teams = pd.read_sql("SELECT ncaa, ncaaid FROM teams", DB.conn)
    def clean_team(s):
        return s.replace(";", "")
    # remove semicolons from some teams
    stacked['team'] = stacked.team.map(lambda s: clean_team(s))
    stacked = stacked.merge(all_teams[['ncaaid', 'ncaa']], left_on="team", right_on="ncaa")
    gb = stacked.groupby('game_id').count()
    doubles = gb[gb.dt == 2].index.values
    stacked = stacked[stacked.game_id.isin(doubles)]
    teams = get_teams(stacked)

    # bteams = pd.read_sql("SELECT * FROM teams WHERE conf='B10'", DB.conn)
    # b10_teams = set(bteams.ncaaid.values)
    # stacked = stacked.loc[(stacked.hteam_id.isin(b10_teams)) & (stacked.ateam_id.isin(b10_teams))]
    # get all the teams in the games dataframe
    keep_columns = ['game_id', 'dt', 'ncaa', 'ncaaid', 'hteam_id', 'ateam_id', 'pts', 'poss']
    stacked = stacked[keep_columns]

    stacked = stacked.merge(teams[['team_id', 'iteam']], left_on='ncaaid', right_on="team_id")

    hgames = stacked[stacked.team_id == stacked.hteam_id]
    agames = stacked[stacked.team_id != stacked.hteam_id]
    games = hgames.merge(agames, on='game_id')
    games = games[['game_id', 'dt_x', 'ncaa_x', 'team_id_x', 'ncaa_y', 'team_id_y', 'pts_x', 'poss_x',
            'pts_y', 'poss_y', 'iteam_x', 'iteam_y']]
    games.rename(columns={'dt_x': 'dt', 'ncaa_x': 'hteam', 'ncaa_y': 'ateam',
                      'ncaaid_x': 'hteam_id', 'ncaaid_y': 'ateam_id', 'pts_x': 'hpts',
                      'pts_y': 'apts', 'poss_x': 'hposs', 'poss_y': 'aposs',
                         'iteam_y': 'i_ateam', 'iteam_x': 'i_hteam'}, inplace=True)
    games['poss'] = games.apply(lambda row: 0.5*(row.hposs + row.aposs), axis=1)
    return games, stacked, teams

if __name__ == "__main__":
    # q = "CREATE VIEW games_detailed AS (%s)" % q1
    # games = get_box_games()#'4/10/2015')
    games, stacked, teams = get_data(2015)
    # stacked = get_data(2015)
