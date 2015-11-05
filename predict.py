import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from DB import DB
import RPI as rpi
import one_bit

def get_season(dt):
    return dt.year if dt.month <= 6 else dt.year + 1

def get_games(season=None):
    games = pd.read_sql("SELECT * FROM games_test", DB.conn)
    # TODO: build this directly into the postgres query
    if season is not None:
        games['season'] = games.dt.map(lambda d: get_season(d))
        games = games[games.season == season]
        games = games.sort('dt')
        games.reset_index(inplace=True)
    return games

def get_teams(games):
    hteams = games[['hteam_id']]
    ateams = games[['ateam_id']]
    hteams = hteams.rename(columns={'hteam_id': 'team_id'})
    ateams = ateams.rename(columns={'ateam_id': 'team_id'})
    teams = pd.concat([hteams, ateams], axis=0)
    teams.drop_duplicates(inplace=True)
    teams.dropna(inplace=True)
    return teams

def train_and_score(X, y):
    clf = LogisticRegression()
    clf.fit(X, y)
    cv = cross_val_score(clf, X, y, cv=5)
    return clf, cv

def group_by_month(df):
    assert 'dt' in df.columns
    df['month'] = df.dt.map(lambda d: d.month)
    df['year'] = df.dt.map(lambda d: d.year)
    return df.groupby(['year', 'month'])

if __name__ =="__main__":
    games = get_games(2015)

    teams = get_teams(games)
    played, wplayed, wins, wwins, games = rpi.rate(teams, games)
    rpi = rpi.get_rpi(played, wplayed, wins, wwins)
    teams['rpi'] = rpi
    teams['gp'] = played.sum(axis=1)

    all_teams = pd.read_sql("SELECT ncaaid, ncaa FROM teams", DB.conn)

    merged_teams = teams.merge(all_teams, how='left', left_on='team_id', right_on='ncaaid')
    print merged_teams.rpi.max(), merged_teams.rpi.min()

    # games = one_bit.rate(games, rating_col='one_bit')
    # outcome = 'home_outcome'
    # features = ['one_bit']
    # games['home_factor'] = np.ones(games.shape[0])
    # X = games[features].values
    # y = games[outcome].values
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2)
    clf, cv = train_and_score(X_train, y_train)
    games['predict'] = clf.predict(X)
    games['error'] = (games.home_outcome.astype(int) - games.predict).abs()




