import pandas as pd
import numpy as np
from DB import DB
import pymc

import util

class Pace(object):

    def __init__(self):
        pass

    def _get_data(self, time_range):
        """
        Get and store game and team data for the given time range.
        :param time_range: A year, date, or string representing the time range of interest.
        :return: None
        """
        self.unstacked, self.stacked, self.teams = util.get_data(time_range)

    def rate(self, time_range, N_samples=10000, burn_rate=0.3):
        """
        Run a markov chain monte carlo simulation using the specified Bayesian network for this
        object's game data.
        :param time_range: A year, date, or string representing the time range of interest.
        :param N_samples: The number of samples to run for the simulation.
        :param burn_rate: A number between 0 and 1 indicating the ratio of samples to discard.
        :return: (PyMC model, PyMC mcmc object)
        """
        assert (burn_rate > 0) and (burn_rate < 1), "burn rate must be between 0 and 1, but was %s" % burn_rate
        self._get_data(time_range)

        num_teams = self.teams.shape[0]
        home_team_idx = self.unstacked.i_hteam.values
        away_team_idx = self.unstacked.i_ateam.values
        observed_pace = self.unstacked.poss.values
        pace_initial = self.stacked.groupby('iteam').mean()['poss'].values
        tau = 1. / pymc.Uniform('sigma', 3, 20)**2
        pace_prior = pymc.Normal("pace_prior", mu=0, tau=tau, size=num_teams, value=pace_initial)
        pace_intercept = pymc.Normal('intercept', np.mean(pace_initial), 1 / 1**2, value=np.mean(pace_initial))

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
        mcmc = pymc.MCMC(model)
        mcmc.sample(N_samples, N_samples * burn_rate)

        return model, mcmc

if __name__ == "__main__":
    pass