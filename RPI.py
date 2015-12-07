import pandas as pd
import numpy as np
from DB import DB
import requests
from bs4 import BeautifulSoup
from datetime import date

import util

class RPIAggregator(object):

    def __init__(self, teams, ignore_nan_teams=True):
        self.ignore_nan_teams = ignore_nan_teams
        self._initialize(teams)

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
        self.col_map = None

    def update(self, game_row):
        """
        Update the aggregator with a single game.

        :param game_row: 1D Numpy Array of the form [hteam_id, ateam_id, home_outcome, neutral].
        :return: None.
        """
        hteam_id = game_row[0]
        ateam_id = game_row[1]
        home_outcome = game_row[2]
        neutral = game_row[3]

        # update winning percentages for teams that don't exist in database
        # TODO: Structure the aggregations to ignore teams not in division 1
        # TODO: so we don't have to worry about nans creeping into aggregation buffers
        if np.isnan(game_row[hteam_id]) or np.isnan(ateam_id):
            if not self.ignore_nan_teams:
                self.update_wp(self.team_index[hteam_id], self.team_index[ateam_id])
            return None
        i, j = self.team_index[hteam_id], self.team_index[ateam_id]

        self._update_wins(i, j, home_outcome, neutral)
        self._update_played(i, j, neutral)

    def merge(self, other):
        pass

    def evaluate(self):
        """

        :return:
        """
        wp = self._calculate_wp()
        owp = self._calculate_owp()
        oowp = self._calculate_oowp(owp)

        return RPIAggregator._calculate_rpi(wp, owp, oowp)

    def _update_played(self, home_idx, away_idx, is_neutral=False):
        """
        Update the total games played aggregation buffers for a single game.

        :param home_idx: INT index of the home team in the aggregation buffer.
        :param away_idx: INT index of the away team in the aggregation buffer.
        :param is_neutral: BOOLEAN indicator of neutral site game.
        :return: None.
        """
        self.weighted_total_played[home_idx] += RPIAggregator.win_factor(True, is_neutral)
        self.weighted_total_played[away_idx] += RPIAggregator.win_factor(False, is_neutral)
        self.total_played[home_idx] += 1
        self.total_played[away_idx] += 1
        self.played[home_idx, away_idx] += 1.
        self.played[away_idx, home_idx] += 1.

    def _update_wins(self, home_idx, away_idx, home_outcome, neutral=False):
        """
        Update the total games won aggregation buffers for a single game.

        :param home_idx: INT index of the home team in the aggregation buffer.
        :param away_idx: INT index of the away team in the aggregation buffer.
        :param home_outcome: BOOLEAN indicator of the home team outcome win or loss.
        :param is_neutral: BOOLEAN indicator of neutral site game.
        :return: None.
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

    def _calculate_wp(self):
        """
        Compute the weighted winning percentage for each team.

        The weighted winning percentage weights home wins less than away wins, and home losses
        more than away losses. Neutral sites are unweighted (weight = 1).

        Reference:
            https://en.wikipedia.org/wiki/Rating_Percentage_Index#Basketball_formula

        :return: 1xnteams Numpy Array of weighted winning percentages.
        """
        return util.safe_divide(self.weighted_total_won, self.weighted_total_played)

    def _calculate_owp(self):
        """
        Compute the opponent's winning percentage for each team.

        The opponent winning percentage for a team is the weighted average of a team's
        opponents unweighted winning percentage. Games that the opponents have played against
        the given team are removed. This is computed as follows, for team A:
            weights * ({team A's opponents wins} - {team A's opponents wins vs. team A}) /
                ({team A's opponents games played} - {team A's opponents games played vs. team A})

        Reference:
            https://en.wikipedia.org/wiki/Rating_Percentage_Index#Basketball_formula

        :return: 1xnteams Numpy Array of opponent's winning percentages.
        """
        opp_wins = np.dot(self.total_won[:, np.newaxis], np.ones(self.nteams)[np.newaxis, :]) - self.wins
        opp_played = np.dot(self.total_played[:, np.newaxis], np.ones(self.nteams)[np.newaxis, :]) - self.played
        weights = util.safe_divide(self.played, self.total_played[:, np.newaxis]).T
        return np.nansum(util.safe_divide(opp_wins, opp_played) * weights, axis=0)

    def _calculate_oowp(self, owp):
        """
        Compute the opponent's opponent's winning percentages for each team.

        The OOWP is the weighted average of a team's OWP. The weights represent the number of times
        the team played each opponent. Games that the opponents have played against
        the given team are removed.

        Reference:
            https://en.wikipedia.org/wiki/Rating_Percentage_Index#Basketball_formula

        :param owp: 1xnteams Numpy Array of opponent's winning percentage for each team.
        :return:
        """
        weights = util.safe_divide(self.played, self.total_played[:, np.newaxis]).T
        return np.nansum(np.dot(owp[:, np.newaxis], np.ones(self.nteams)[np.newaxis, :]) * weights, axis=0)

    @staticmethod
    def win_factor(is_home, is_neutral=False):
        if is_neutral:
            return 1.
        elif is_home:
            return 0.6
        else:
            return 1.4

    @staticmethod
    def _calculate_rpi(wp, owp, oowp):
        return 0.25 * wp + 0.5 * owp + 0.25 * oowp

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

if __name__ == "__main__":
    df = rpi_test_data()
    teams = util.get_teams(df)
    data = df[['hteam_id', 'ateam_id', 'home_outcome', 'neutral']].values
    agg = RPIAggregator(teams)
    map(lambda x: agg.update(x), data)

