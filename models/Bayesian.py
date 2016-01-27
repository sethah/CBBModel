import pandas as pd
import numpy as np
from DB import DB
import pymc

from RatingsModel import RatingsModel

import util

class BayesianSkill(object):

    def __init__(self):
        pass

    def _initial_guess(self, attribute):
        if attribute == 'pace':
            return self.stacked.groupby('iteam').mean()['poss'].values

    def rate(self, time_range, N_samples=10000, burn_rate=0.3):
        """
        TODO
        """
        assert (burn_rate >= 0) and (burn_rate <= 1), "burn rate must be between 0 and 1, but was %s" % burn_rate
        self._get_data(time_range)
        num_teams = self.teams.shape[0]
        home_team_idx = self.unstacked.i_hteam.values
        away_team_idx = self.unstacked.i_ateam.values

        tau_oskills = pymc.Normal('tau_oskills', 1, 2)
        oskills_prior = pymc.Normal("oskills", mu=0, tau=tau_oskills, size=num_teams, value=self._initial_guess())
        tau_dskills = pymc.Normal('tau_dskills', 1, 2)
        dskills_prior = pymc.Normal("dskills", mu=0, tau=tau_oskills, size=num_teams, value=self._initial_guess())
        home_advantage = pymc.Normal('home', 0.0, 0.001, value=0)
        intercept = pymc.Normal('ppp_intercept', 1.0, 0.001)

        @pymc.deterministic()
        def oskill_rtg(oskill=oskills_prior):
            o = oskill.copy()
            o = o - np.mean(oskill)
            return o

        @pymc.deterministic()
        def dskill_rtg(dskill=dskills_prior):
            d = dskill.copy()
            d = d - np.mean(dskill)
            return d

        @pymc.deterministic()
        def home_ppp(hteam=home_team_idx,
                     ateam=away_team_idx,
                     home_advantage=home_advantage,
                     oskills=oskill_rtg,
                     dskills=dskill_rtg,
                     intercept=intercept):
            return intercept + home_advantage + oskills[hteam] + dskills[ateam]

        @pymc.deterministic()
        def away_ppp(hteam=home_team_idx,
                     ateam=away_team_idx,
                     oskills=oskill_rtg,
                     dskills=dskill_rtg,
                     intercept=intercept):
            return intercept + oskills[ateam] + dskills[hteam]

        model = pymc.Model([mu_pace, pace_prior, tau, pace_rtg, poss, pace_intercept, tau_poss, poss_pred])
        # map_ = pymc.MAP(model)
        # map_.fit(method='fmin_powell')
        mcmc = pymc.MCMC(model)
        mcmc.sample(N_samples, N_samples * burn_rate)

        return model, mcmc

class Pace(RatingsModel):

    def __init__(self, **kwargs):
        self.set_params(**self._default_params())
        self.set_params(**kwargs)

    def _default_params(self):
        return {'n_samples': 10000, 'burn_rate': 0.3}

    def _initial_guess(self):
        return self.stacked.groupby('iteam').mean()['poss'].values

    def rate(self, unstacked):
        """
        Run a Markov Chain Monte Carlo (MCMC) simulation on the defined
        directed graphical model (aka Bayesian Network).

        References
        ----------
        TODO

        Parameters
        ----------
        unstacked : dataframe
            Unstacked dataframe containing game and stat information.

        Returns
        -------
        TODO
        """
        util.validate_games(unstacked, ['poss'])
        assert (self.burn_rate >= 0) and (self.burn_rate <= 1), \
            "burn rate must be between 0 and 1, but was %s" % self.burn_rate

        unstacked = unstacked.sort('dt')
        teams = Pace._get_teams(unstacked)

        num_teams = teams.shape[0]
        home_team_idx = unstacked.i_hteam.values
        away_team_idx = unstacked.i_ateam.values
        observed_pace = unstacked.poss.values
        pace_initial = self._initial_guess()
        # tau = 1. / pymc.Uniform('sigma', 3, 20)**2
        tau = pymc.Uniform('tau', 1. / 40**2, 1. / 20**2)
        pace_prior = pymc.Normal("pace_prior", mu=0, tau=tau, size=num_teams, value=pace_initial)
        pace_intercept = pymc.Normal('intercept', 66, 1 / 1**2, value=66)

        @pymc.deterministic
        def pace_rtg(pace=pace_prior):
            p = pace.copy()
            p = p - np.mean(pace)
            return p

        @pymc.deterministic
        def mu_pace(home_team=home_team_idx, away_team=away_team_idx,
                    paces=pace_rtg, pace_intercept=pace_intercept):
            return pace_intercept + paces[home_team] + paces[away_team]

        tau_poss = 1. / pymc.Uniform('sigma_poss', 1., 10.)**2
        poss = pymc.Normal('poss', mu=mu_pace, tau=tau_poss, value=observed_pace, observed=True)
        poss_pred = pymc.Normal('poss_pred', mu=mu_pace, tau=tau_poss)
        model = pymc.Model([mu_pace, pace_prior, tau, pace_rtg, poss, pace_intercept, tau_poss, poss_pred])
        # map_ = pymc.MAP(model)
        # map_.fit(method='fmin_powell')
        mcmc = pymc.MCMC(model)
        mcmc.sample(self.n_samples, self.n_samples * self.burn_rate)

        return model, mcmc

if __name__ == "__main__":
    pass