import pandas as pd
import numpy as np
from DB import DB
import requests
from bs4 import BeautifulSoup
from datetime import date

import util

def safe_divide(num, den):
    return np.nan_to_num(num / den)

def rpi_test_data():
    team_ids = {'UConn': 0, 'Kansas': 1, 'Duke': 2, 'Minnesota': 3}
    data = [['UConn', 64, 'Kansas', 57],
            ['UConn', 82, 'Duke', 68],
            ['Minnesota', 71, 'UConn', 72],
            ['Kansas',	69,	'UConn', 62],
            ['Duke', 81, 'Minnesota', 70],
            ['Minnesota', 52, 'Kansas', 62]]
    df = pd.DataFrame(data, columns=['hteam', 'hscore', 'ateam', 'ascore'])
    df['home_outcome'] = df.hscore > df.ascore
    df['neutral'] = False
    df['hteam_id'] = df.hteam.map(lambda x: team_ids.get(x))
    df['ateam_id'] = df.ateam.map(lambda x: team_ids.get(x))
    return df

def test_rpi():
    df = rpi_test_data()
    agg = RPIAggregator()
    agg.rate_for_games(df)

class RPIAggregator(object):

    def __init__(self, ignore_nan_teams=True):
        self.ignore_nan_teams = ignore_nan_teams
        self.initialized = False

    def _initialize(self, teams):
        """
        Allocate arrays for data storage during aggregation.

        :param teams: A dataframe of team ids
        :return: None
        """
        self.played = np.zeros(shape=(teams.shape[0], teams.shape[0]))
        self.wins = np.zeros(shape=(teams.shape[0], teams.shape[0]))
        self.team_index = {int(team) if not np.isnan(team) else -1: idx for idx, team in enumerate(teams.team_id.values)}
        self.inverted_team_index = {v: k for k, v in self.team_index.items()}
        self.weighted_total_won = np.zeros(teams.shape[0])
        self.weighted_total_played = np.zeros(teams.shape[0])
        self.total_played = np.zeros(teams.shape[0])
        self.total_won = np.zeros(teams.shape[0])
        self.wp = np.zeros(teams.shape[0])
        self.owp = np.zeros(teams.shape[0])
        self.oowp = np.zeros(teams.shape[0])
        self.nteams = len(teams)
        self.initialized = True

    def infinite_depth(self):
        """
        Calculate infinite depth RPI via an iterative algorithm

        :return: 1D Numpy array of RPI ratings for each team.
        """
        rpi = RPIAggregator.calculate_rpi(self.wp, self.owp, self.oowp)
        converged = False
        i = 0
        while not converged:
            prev_rpi = rpi
            rpi_mat = np.dot(rpi[:,np.newaxis], np.ones(rpi.shape[0])[:,np.newaxis].T).T

            rpi = self.wp - 0.5 + np.average(rpi_mat, weights=self.played, axis=1)
            resid = np.linalg.norm(rpi - prev_rpi)
            print resid
            i += 1
            converged = i > 10

        return rpi

    def get_update_indices(self, home_idx, away_idx):
        """

        :param home_idx:
        :param away_idx:
        :return:
        """
        home_opponent_indices = np.nonzero(self.played[home_idx, :])[0]
        away_opponent_indices = np.nonzero(self.played[away_idx, :])[0]
        return np.hstack((home_opponent_indices,
                          away_opponent_indices,
                          [home_idx, away_idx]))

    def update_wp(self, home_idx, away_idx):
        """
        Update the weighted win percentages for home/away teams for RPI calculation.

        :param home_idx: INT - index of home team in the stats arrays.
        :param away_idx: INT - index of away team in the stats arrays.
        :return: None
        """
        if not np.isnan(home_idx):
            hwp = safe_divide(self.weighted_total_won[home_idx], self.weighted_total_played[home_idx])
            self.wp[home_idx] = hwp
        if not np.isnan(away_idx):
            awp = safe_divide(self.weighted_total_won[away_idx], self.weighted_total_played[away_idx])
            self.wp[away_idx] = awp

    def update_owp(self, home_idx, away_idx):
        """

        :param home_idx:
        :param away_idx:
        :return:
        """
        update_indices = self.get_update_indices(home_idx, away_idx)

        for idx in update_indices:
            opp_wins = self.get_opp_wins(idx)
            opp_played = self.get_opp_played(idx)
            weights = self.get_weights(idx)
            self.owp[idx] = np.nansum(safe_divide(opp_wins, opp_played) * weights)

    def _calculate_wp(self):
        """

        :return:
        """
        return safe_divide(self.weighted_total_won, self.weighted_total_played)

    def _calculate_owp(self):
        """

        :return:
        """
        opp_wins = np.dot(self.total_won[:, np.newaxis], np.ones(self.nteams)[np.newaxis, :]) - self.wins
        opp_played = np.dot(self.total_played[:, np.newaxis], np.ones(self.nteams)[np.newaxis, :]) - self.played
        weights = safe_divide(self.played, self.total_played[:, np.newaxis]).T
        return np.nansum(safe_divide(opp_wins, opp_played) * weights, axis=0)

    def _calculate_oowp(self, owp):
        """

        :param owp:
        :return:
        """
        weights = safe_divide(self.played, self.total_played[:, np.newaxis]).T
        return np.nansum(np.dot(owp[:, np.newaxis], np.ones(self.nteams)[np.newaxis, :]) * weights, axis=0)

    def update_oowp(self, home_idx, away_idx):
        """

        :param home_idx:
        :param away_idx:
        :return:
        """
        home_oowp = np.nansum(self.owp * self.get_weights(home_idx))
        away_oowp = np.nansum(self.owp * self.get_weights(away_idx))
        self.oowp[home_idx] = home_oowp
        self.oowp[away_idx] = away_oowp

    def get_weights(self, idx):
        """

        :param idx:
        :return:
        """
        return safe_divide(self.played[idx, :], self.total_played[idx])

    def get_opp_wins(self, idx):
        """

        :param idx:
        :return:
        """
        # the total games won by each team minus the games each team has won against
        # a particular team
        return self.total_won - self.wins[:, idx]

    def get_opp_played(self, idx):
        """

        :param idx:
        :return:
        """
        # the total games played by each team minus the games each team has played against
        # a particular team
        return self.total_played - self.played[:, idx]

    def get_rpi(self, idx):
        """

        :param idx:
        :return:
        """
        return RPIAggregator._calculate_rpi(self.wp[idx], self.owp[idx], self.oowp[idx])

    def calculate(self):
        """

        :return:
        """
        wp = self._calculate_wp()
        owp = self._calculate_owp()
        oowp = self._calculate_oowp(owp)
        print wp, owp, oowp

        return RPIAggregator._calculate_rpi(wp, owp, oowp)

    def rate_at_date(self, dt):
        """

        :param dt:
        :return:
        """
        games = util.get_games(dt)
        self.rate_for_games(games)

    def rate_for_games(self, games):
        """

        :param games:
        :return:
        """
        self.teams = util.get_teams(games)
        self._initialize(self.teams)
        cols = {name: idx for idx, name in enumerate(games.columns)}
        for k, row in enumerate(games.values):
            hteam_id = row[cols['hteam_id']]
            ateam_id = row[cols['ateam_id']]

            # update winning percentages for teams that don't exist in database
            if np.isnan(row[cols['hteam_id']]) or np.isnan(row[cols['ateam_id']]):
                if not self.ignore_nan_teams:
                    self.update_wp(self.team_index[hteam_id], self.team_index[ateam_id])
                continue
            i, j = self.team_index[hteam_id], self.team_index[ateam_id]

            self.update_wins(i, j, row[cols['home_outcome']], row[cols['neutral']])
            self.update_played(i, j, row[cols['neutral']])


    def rate_for_every_game(self, teams, games, store_intermediate=False):
        """

        :param teams:
        :param games:
        :param store_intermediate:
        :return:
        """
        self._initialize(teams)
        home_rpi = np.zeros(games.shape[0])
        away_rpi = np.zeros(games.shape[0])
        hwp = np.zeros(games.shape[0])
        howp = np.zeros(games.shape[0])
        hoowp = np.zeros(games.shape[0])
        awp = np.zeros(games.shape[0])
        aowp = np.zeros(games.shape[0])
        aoowp = np.zeros(games.shape[0])
        home_gp = np.zeros(games.shape[0])
        away_gp = np.zeros(games.shape[0])
        cols = {name: idx for idx, name in enumerate(games.columns)}
        for k, row in enumerate(games.values):
            hteam_id = row[cols['hteam_id']]
            ateam_id = row[cols['ateam_id']]

            if np.isnan(row[cols['hteam_id']]) or np.isnan(row[cols['ateam_id']]):
                if not self.ignore_nan_teams:
                    self.update_wp(self.team_index[hteam_id], self.team_index[ateam_id])
                continue
            i, j = self.team_index[hteam_id], self.team_index[ateam_id]

            self.update_wp(i, j)
            self.update_owp(i, j)
            self.update_oowp(i, j)
            hwp[k] = self.wp[i]
            howp[k] = self.owp[i]
            hoowp[k] = self.oowp[i]
            awp[k] = self.wp[j]
            aowp[k] = self.owp[j]
            aoowp[k] = self.oowp[j]
            home_rpi[k] = self.get_rpi(i)
            away_rpi[k] = self.get_rpi(j)
            home_gp[k] = self.total_played[i]
            away_gp[k] = self.total_played[j]

            self.update_wins(i, j, row[cols['home_outcome']], row[cols['neutral']])
            self.update_played(i, j, row[cols['neutral']])

        return {'home_rpi': home_rpi, 'away_rpi': away_rpi, 'hwp': hwp, 'howp': howp, 'hoowp': hoowp,
                'awp': awp, 'aowp': aowp, 'aoowp': aoowp}

    def update_played(self, home_idx, away_idx, is_neutral=False):
        """
        Update the total games played for a
        :param home_idx:
        :param away_idx:
        :param is_neutral:
        :return:
        """
        self.weighted_total_played[home_idx] += RPIAggregator.win_factor(True, is_neutral)
        self.weighted_total_played[away_idx] += RPIAggregator.win_factor(False, is_neutral)
        self.total_played[home_idx] += 1
        self.total_played[away_idx] += 1
        self.played[home_idx, away_idx] += 1.
        self.played[away_idx, home_idx] += 1.

    def update_wins(self, home_idx, away_idx, home_outcome, neutral=False):
        """

        :param home_idx:
        :param away_idx:
        :param home_outcome:
        :param neutral:
        :return:
        """
        win_factor = RPIAggregator.win_factor(home_outcome, neutral)
        if home_outcome:
            self.weighted_total_won[home_idx] += win_factor
            self.total_won[home_idx] += 1
            self.wins[home_idx, away_idx] += 1.
        else:
            self.weighted_total_won[away_idx] += win_factor
            self.total_won[away_idx] += 1
            self.wins[away_idx, home_idx] += 1.

    def pretty_rating(self):
        """

        :return:
        """
        ratings = self.calculate()
        ratings_with_teams = [[rating, self.inverted_team_index[i]] for i, rating in enumerate(ratings)]
        df = pd.DataFrame(ratings_with_teams, columns=['rating', 'team_id'])
        teams = pd.read_sql("SELECT ncaaid, ncaa FROM teams", DB.conn)
        return df.merge(teams, left_on="team_id", right_on="ncaaid")

    @staticmethod
    def _calculate_rpi(wp, owp, oowp):
        return 0.25 * wp + 0.5 * owp + 0.25 * oowp

    @staticmethod
    def win_factor(is_home, is_neutral=False):
        if is_neutral:
            return 1.
        elif is_home:
            return 0.6
        else:
            return 1.4

def get_true_rpi():
    url = 'http://www.cbssports.com/collegebasketball/bracketology/nitty-gritty-report'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    rpi_table = soup.find('table', {'class': 'ncaa-rankings-table'})


if __name__ == "__main__":
    # df = rpi_test_data()
    # agg = RPIAggregator(True)
    #
    # agg.rate_at_date(date(2015, 03, 20))
#     url = 'http://www.cbssports.com/collegebasketball/bracketology/nitty-gritty-report'
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, "html.parser")
#     rpi_table = soup.find('table', {'class': 'data'})
#     team_links = rpi_table.findAll('a')
#     cbs_teams = []
#     for link in team_links:
#         url = link['href']
#         if '/teams/' not in url:
#             continue
#         team = url.split('/')[-2]
#         cbs_teams.append(team)
#     cbsdf = pd.DataFrame(np.unique(cbs_teams), columns=['team'])
#     df = pd.read_html(str(rpi_table), header=2)[0]
#     df['cbs1'] = cbs_teams
#     teams = pd.read_sql("SELECT * FROM teams", DB.conn)
    # pass
    games = util.get_games(2015)
    teams = util.get_teams(games)
    agg = RPIAggregator()
    season_dict = agg.rate_for_every_game(teams, games)
    # rpi = agg.infinite_depth()
