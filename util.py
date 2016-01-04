import pandas as pd
import numpy as np
import datetime
from scipy import spatial

from DB import DB
import org_ncaa

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

def get_teams(stacked, min_games=0, input_col='team', output_col='team', gb_col='home_outcome'):
    """
    Create a dataframe of all teams in a dataframe of games, filtering out teams with
    less than `min_games`.
    :param stacked: DATAFRAME - stacked game data.
    :param min_games: INT - minimum number of games played for each team.
    :param input_col: STRING - name of team column in the games dataframe.
    :param output_col: STRING - name of team column in the new dataframe.
    :param gb_col: STRING - name of column to use in groupby count.
    :return: DATAFRAME - unique set of teams
    """
    gb = stacked.groupby(input_col).count()
    gb = gb[gb[gb_col] > min_games]
    teams = pd.DataFrame(gb.index.values, columns=[output_col])
    teams['iteam'] = teams.index.values
    return teams

def stack(unstacked, col, left_col, join_cols):
    assert unstacked.shape[0] % 2 == 0, "unstacked data frame should have an even number of rows"
    left = unstacked[unstacked[col] == unstacked[left_col]]
    right = unstacked[unstacked[col] != unstacked[left_col]]
    stacked = left.merge(right, on=join_cols)
    return stacked

def unstack(stacked, swap_left, swap_right, rename_left=None, rename_right=None):
    stacked['home_team'] = stacked['hteam_id']
    left = stacked
    right = stacked.rename(columns={swap_left: swap_right, swap_right: swap_left})
    unstacked = pd.concat([left, right])
    if rename_left:
        assert rename_right, "must provide renamed columns for each stacking column"
        unstacked.rename(columns={swap_left: rename_left, swap_right: rename_right,
                                  'hteam': 'team', 'ateam': 'opp'}, inplace=True)
    return unstacked

def query_stat(stat):
    d = {'pts': 'b.pts',
         'poss': 'b.fga - COALESCE(b.oreb, 0) + COALESCE(b.turnover, 0) + 0.475 * COALESCE(b.fta, 0) AS poss'}
    return d.get(stat)

def get_team_stats(time_range=None, stats=None):
    """
    Get a stacked (two entries per game) of game summary and box stats data
    at a team level.
    :param time_range: time range to filter by
    :param stats: LIST - stats to collect in the query
    :return: DATAFRAME - stacked dataframe of game data
    """
    bounds = get_date_bounds(time_range)
    queries = filter(lambda x: x, [query_stat(stat) for stat in stats])
    stat_query = ', '.join(queries)
    q = """SELECT
                g.*,
                b.team,
                %s
            FROM box_test b
            JOIN games_test g
            ON b.game_id = g.game_id
            AND b.first_name='Totals'
            AND g.dt > '%s'
            AND g.dt <= '%s'
        """ % (stat_query, bounds[0], bounds[1])
    df = pd.read_sql(q, DB.conn).dropna(subset=['ateam_id', 'hteam_id'])
    df['team'] = df.team.map(lambda s: s.replace(';', ''))
    df['team'] = df.team.map(lambda t: org_ncaa.REVERSE_ALIAS.get(t, t))
    return df

def filter_teams(stacked):
    """
    Filter out games where one or more teams aren't in the teams database.
    :param stacked: DATAFRAME - stacked dataframe of games and stats.
    :return: DATAFRAME - filtered, stacked dataframe.
    """
    all_teams = pd.read_sql("SELECT ncaa, ncaaid AS team_id FROM teams", DB.conn)
    stacked = stacked.merge(all_teams, left_on="team", right_on="ncaa")
    stacked.drop('ncaa', axis=1, inplace=True)
    gb = stacked.groupby('game_id').count()
    doubles = gb[gb.dt == 2].index.values
    return stacked[stacked.game_id.isin(doubles)]

def get_data(time_range=None):
    """
    Get games data for a given time interval.
    :param time_range: A year, date, or string representing the time interval of interest.
    :return: (DATAFRAME, DATAFRAME, DATAFRAME)
    """
    stacked = get_team_stats(time_range, ['pts', 'poss'])
    stacked = filter_teams(stacked)
    teams = get_teams(stacked, input_col='team_id', output_col='team_id')

    # join to get the team index as a column
    stacked = stacked.merge(teams[['team_id', 'iteam']], on="team_id")

    hgames = stacked[stacked.team_id == stacked.hteam_id]
    agames = stacked[stacked.team_id != stacked.hteam_id]
    agames = agames[['game_id', 'team', 'pts', 'poss', 'team_id', 'iteam']]
    unstacked = hgames.merge(agames, on='game_id')
    unstacked = unstacked[['game_id', 'dt', 'team_x', 'team_id_x', 'team_y', 'team_id_y',
                           'pts_x', 'poss_x', 'pts_y', 'poss_y', 'iteam_x', 'iteam_y']]
    unstacked.rename(columns={'team_x': 'hteam', 'team_y': 'ateam', 'team_id_x': 'hteam_id',
                              'team_id_y': 'ateam_id', 'pts_x': 'hpts', 'pts_y': 'apts',
                              'poss_x': 'hposs', 'poss_y': 'aposs', 'iteam_y': 'i_ateam',
                              'iteam_x': 'i_hteam'}, inplace=True)
    unstacked['poss'] = unstacked.apply(lambda row: 0.5*(row.hposs + row.aposs), axis=1)
    return unstacked, stacked, teams

def compare_ratings(this, that, compare_col, this_name='this', that_name='that', join_col='team_id', metric=None):
    if this.shape[0] > that.shape[0]:
        df = that.merge(this, on=join_col)
    else:
        df = this.merge(that, on=join_col)
    df.rename(columns={compare_col + '_x': this_name + '_rating',
                       compare_col + '_y': that_name + '_rating'}, inplace=True)
    df[this_name + '_rank'] = df[this_name + '_rating'].rank(ascending=False)
    df[that_name + '_rank'] = df[that_name + '_rating'].rank(ascending=False)
    if metric is None:
        _metric = 'cos'
    else:
        _metric = metric
    rank_sim = _ranking_similarity(df[this_name + '_rank'].values, df[that_name + '_rank'].values, _metric)
    rating_sim = None

    return rank_sim, rating_sim, df


def _ranking_similarity(rank1, rank2, metric='cos'):
    print metric
    assert rank1.shape[0] == rank2.shape[0], 'rank vectors must be of same length'
    if metric == 'avg_diff':
        sim = np.mean(np.abs(rank1 - rank2))
    elif metric == 'cos':
        sim = spatial.distance.cosine(rank1, rank2)
    else:
        sim = None

    return sim

if __name__ == "__main__":
    unstacked, stacked, teams = get_data(2015)
