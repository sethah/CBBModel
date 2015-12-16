import numpy as np
import pandas as pd
import numpy as np
from DB import DB
import requests
from bs4 import BeautifulSoup
from datetime import date
import pymc

import util

class OffenseDefense(object):

    def __init__(self):
        pass

    def get_teams(self, games):
        hteams = games[['hteam_id']]
        ateams = games[['ateam_id']]
        hteams = hteams.rename(columns={'hteam_id': 'team_id'})
        ateams = ateams.rename(columns={'ateam_id': 'team_id'})
        teams = pd.concat([hteams, ateams], axis=0)
        teams.drop_duplicates(inplace=True)
        teams = teams.reset_index()
        teams['iteam'] = teams.index.values
        teams.drop('index', 1, inplace=True)
        return teams

    def stack_games(self, games, teams):
        games = games.merge(teams, left_on='hteam_id', right_on='team_id', how='left')
        games = games.rename(columns={'iteam': 'i_home'})
        games.drop('team_id', 1, inplace=True)
        games = games.merge(teams, left_on='ateam_id', right_on='team_id', how='left')
        games = games.rename(columns={'iteam': 'i_away'})
        games.drop('team_id', 1, inplace=True)

        return games

    def get_starting_points(self, stacked_games):
        averages = stacked_games.groupby('hteam_id').mean()
        off_starting_points = np.log(averages['home_score'].values)
        def_starting_points = -np.log(averages['away_score'].values)
        return off_starting_points, def_starting_points

    def algo(self, teams, games):
        # find mu and sigma that maximize outcomes given scores
        stacked = self.stack_games(games, teams)
        num_teams = teams.shape[0]

        observed_home_points = stacked.home_score.values
        observed_away_points = stacked.away_score.values

        o_initial, d_initial = self.get_starting_points(stacked)

        home_team_idx = stacked.i_home.values
        away_team_idx = stacked.i_away.values
        home_prior_mean = 0.
        home_prior_std = 10.
        home = pymc.Normal('home', home_prior_mean, 1 / home_prior_std**2, value=0)
        # tau = 1 / stddev^2
        tau_off = pymc.Uniform('tau_off', 0.0, 1.0, )  # equivalent to stddev between 1 and inf
        tau_def = pymc.Uniform('tau_def', 0.0, 1.0, )  # equivalent to stddev between 1 and inf
        intercept = pymc.Normal('intercept', 0, 1 / (100.)**2, value=0)

        # prior for offensive skills is a gaussian for each team
        o_star = pymc.Normal("o_star", mu=0, tau=tau_off, size=num_teams, value=o_initial)
        d_star = pymc.Normal("d_star", mu=0, tau=tau_def, size=num_teams, value=d_initial)

        # trick to code the sum to zero contraint
        @pymc.deterministic
        def o_rtg(offs_star=o_star):
            offs = offs_star.copy()
            offs = offs - np.mean(offs_star)
            return offs

        @pymc.deterministic
        def d_rtg(defs_star=d_star):
            defs = defs_star.copy()
            defs = defs - np.mean(defs_star)
            return defs

        @pymc.deterministic
        def home_theta(home_team=home_team_idx,
                       away_team=away_team_idx,
                       home=home,
                       offs=o_rtg,
                       defs=d_rtg,
                      intercept=intercept):
            return np.exp(intercept + home + offs[home_team] + defs[away_team])

        @pymc.deterministic
        def away_theta(home_team=home_team_idx,
                       away_team=away_team_idx,
                       offs=o_rtg,
                       defs=d_rtg,
                      intercept=intercept):
            return np.exp(intercept + offs[away_team] + defs[home_team])


        home_points = pymc.Poisson('home_points',
                                  mu=home_theta,
                                  value=observed_home_points,
                                  observed=True)
        away_points = pymc.Poisson('away_points',
                                  mu=away_theta,
                                  value=observed_away_points,
                                  observed=True)

        model = pymc.Model([home, intercept, tau_off, tau_def,
                          home_theta, away_theta,
                          o_star, d_star, o_rtg, d_rtg,
                          home_points, away_points])
        mcmc = pymc.MCMC(model)
        mcmc.sample(10000, 3000)
        return o_rtg.stats(), d_rtg.stats(), home, home_theta, intercept

if __name__ == "__main__":
    games = util.get_box_games(2015)
    bteams = pd.read_sql("SELECT * FROM teams WHERE conf='B10'", DB.conn)
    b10_teams = set(bteams.ncaaid.values)
    b10 = games[(games.hteam_id.isin(b10_teams)) & (games.ateam_id.isin(b10_teams))]
    b10['winner'] = b10.apply(lambda row: row.hteam_id if row.home_outcome else row.ateam_id, axis=1)
    agg = OffenseDefense()
    teams = agg.get_teams(b10)
    teams = teams.merge(bteams[['ncaaid', 'ncaa']], left_on='team_id', right_on='ncaaid')
    o, d, home, theta, i = agg.algo(teams, b10)
    teams['ortg'] = o['mean']
    teams['drtg'] = d['mean']
    os, ds = agg.get_starting_points(agg.stack_games(b10, teams))