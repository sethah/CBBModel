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

    def possession_model(self, date_like):
        games, stacked, teams = util.get_data(date_like)
        pace_initial = stacked.groupby('iteam').mean()['poss'].values
        teams['pace_initial'] = pace_initial
        num_teams = teams.shape[0]
        home_team_idx = games.i_hteam.values
        away_team_idx = games.i_ateam.values
        observed_pace = games.poss.values
        tau = 1. / pymc.Uniform('sigma', 3, 20)**2
        pace_prior = pymc.Normal("pace_prior", mu=0, tau=tau, size=num_teams, value=pace_initial)
        pace_intercept = pymc.Normal('intercept', 4, 1 / (1)**2, value=4)

        @pymc.deterministic
        def pace_rtg(pace=pace_prior):
            p = pace.copy()
            p = p - np.mean(pace)
            return p

        @pymc.deterministic
        def mu_pace(home_team=home_team_idx,
                       away_team=away_team_idx,
                       paces=pace_rtg,
                       pace_intercept=pace_intercept):
            return pace_intercept + paces[home_team] + paces[away_team]

        tau_poss = 1. / pymc.Uniform('sigma_poss', 1, 10)**2
        poss = pymc.Normal('poss', mu=mu_pace, tau=tau_poss, value=observed_pace, observed=True)
        poss_pred = pymc.Normal('poss_pred', mu=mu_pace, tau=tau_poss)

        model = pymc.Model([mu_pace, pace_prior, tau, pace_rtg, poss, pace_intercept, tau_poss, poss_pred])
        mcmc = pymc.MCMC(model)
        N = 10000
        burn_in = 3000
        mcmc.sample(N, burn_in)
        teams['mean_pace_rtg'] = pace_rtg.stats()['mean']
        teams['pace_rtg_rank'] = teams.mean_pace_rtg.rank(ascending=False)

        return model


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
        eff_intercept = pymc.Normal('intercept', 0, 1 / (1.)**2, value=0)
        pace_intercept = pymc.Normal('intercept', 5, 1 / (3.)**2, value=0)

        # prior for offensive skills is a gaussian for each team
        o_star = pymc.Normal("o_star", mu=0, tau=tau_off, size=num_teams, value=o_initial)
        d_star = pymc.Normal("d_star", mu=0, tau=tau_def, size=num_teams, value=d_initial)
        o_pace_star = pymc.Normal("o_pace_star", mu=0, tau=tau_off, size=num_teams, value=o_initial)
        d_pace_star = pymc.Normal("d_pace_star", mu=0, tau=tau_def, size=num_teams, value=d_initial)

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
        def o_pace(offs_star=o_pace_star):
            offs = offs_star.copy()
            offs = offs - np.mean(offs_star)
            return offs

        @pymc.deterministic
        def d_pace(defs_star=d_pace_star):
            defs = defs_star.copy()
            defs = defs - np.mean(defs_star)
            return defs

        @pymc.deterministic
        def home_eff(home_team=home_team_idx,
                       away_team=away_team_idx,
                       home=home,
                       off_effs=o_rtg,
                       def_effs=d_rtg,
                       eff_intercept=eff_intercept):
            return eff_intercept + home + off_effs[home_team] + def_effs[away_team]

        @pymc.deterministic
        def home_pace(home_team=home_team_idx,
                       away_team=away_team_idx,
                       offs=o_pace,
                       defs=d_pace,
                       eff_intercept=eff_intercept):
            return eff_intercept + offs[home_team] + defs[away_team]

        @pymc.deterministic
        def home_theta(eff=home_eff, pace=home_pace):
            return np.exp(eff * pace)

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