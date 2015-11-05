import pandas as pd
import numpy as np
from DB import DB
import time

def home_win_factor(neutral=False):
    return 1. if neutral else 0.6
def away_win_factor(neutral=False):
    return 1. if neutral else 1.4

def safe_divide(num, den):
    return np.nan_to_num(num / den)

def rate(teams, games, n_games=1000000):
    team_index = {int(team) if not np.isnan(team) else -1: idx for (team, idx) in \
                  zip(teams.team_id.values, range(teams.shape[0]))}
    # home_win_factor = 0.6
    # away_win_factor = 1.4

    wplayed = np.zeros(shape=(teams.shape[0], teams.shape[0]))
    played = np.zeros(shape=(teams.shape[0], teams.shape[0]))
    wwins = np.zeros(shape=(teams.shape[0], teams.shape[0]))
    wins = np.zeros(shape=(teams.shape[0], teams.shape[0]))
    home_rpi = np.zeros(games.shape[0])
    away_rpi = np.zeros(games.shape[0])
    gp = np.zeros(teams.shape[0])
    home_gp = np.zeros(games.shape[0])
    away_gp = np.zeros(games.shape[0])
    # print games.shape, len(team_index)
    for k, (idx, game) in enumerate(games.iterrows()):
        # TODO: handle nan teams
        if np.isnan(game.hteam_id) or np.isnan(game.ateam_id):
            # print k
            continue
        i, j = (team_index[game.hteam_id], team_index[game.ateam_id])
        wp = win_percentage(wwins, wplayed)
        unweighted_owp = top_level_owp(wins, played)
        owp = arbitrary_owp(unweighted_owp, played)
        oowp = arbitrary_owp(owp, played)
        # print wp[j], owp[j], oowp[j]
        rpi = get_rpi(played, wplayed, wins, wwins)
        home_rpi[k] = rpi[i]
        away_rpi[k] = rpi[j]
        print rpi[i], rpi[j]
        gp[i] += 1
        gp[j] += 1
        home_gp[k] = gp[i]
        away_gp[k] = gp[j]
        # print game['dt'], rpi[i], rpi[j], home_rpi[k], away_rpi[k]
        # teams['rpi'] = rpi

        if game.home_outcome:
            wwins[i, j] += home_win_factor(game.neutral)
            wins[i, j] += 1.
        else:
            wwins[j, i] += away_win_factor(game.neutral)
            wins[j, i] += 1.
        wplayed[i, j] += home_win_factor(game.neutral)
        wplayed[j, i] += away_win_factor(game.neutral)
        played[i, j] += 1.
        played[j, i] += 1.

    games['home_rpi'] = home_rpi
    games['away_rpi'] = away_rpi
    games['home_gp'] = home_gp
    games['away_gp'] = away_gp

    return played, wplayed, wins, wwins, games

class RPIAggregator(object):

    def __init__(self, teams):
        self.played = np.zeros(shape=(teams.shape[0], teams.shape[0]))
        self.wins = np.zeros(shape=(teams.shape[0], teams.shape[0]))
        self.team_index = {int(team) if not np.isnan(team) else -1: idx for idx, team in enumerate(teams.team_id.values)}
        self.weighted_total_won = np.zeros(teams.shape[0])
        self.weighted_total_played = np.zeros(teams.shape[0])
        self.total_played = np.zeros(teams.shape[0])
        self.total_won = np.zeros(teams.shape[0])
        self.wp = np.zeros(teams.shape[0])
        self.owp = np.zeros(teams.shape[0])
        self.oowp = np.zeros(teams.shape[0])

    def update_wp(self, home_idx, away_idx):
        hwp = safe_divide(self.weighted_total_won[home_idx], self.weighted_total_played[home_idx])
        awp = safe_divide(self.weighted_total_won[away_idx], self.weighted_total_played[away_idx])
        self.wp[home_idx] = hwp
        self.wp[away_idx] = awp

    def get_update_indices(self, home_idx, away_idx):
        home_opponent_indices = np.nonzero(self.played[home_idx, :])[0]
        away_opponent_indices = np.nonzero(self.played[away_idx, :])[0]
        return np.hstack((home_opponent_indices,
                          away_opponent_indices,
                          [home_idx, away_idx]))

    def update_owp(self, home_idx, away_idx):
        update_indices = self.get_update_indices(home_idx, away_idx)

        for idx in update_indices:
            opp_wins = self.get_opp_wins(idx)
            opp_played = self.get_opp_played(idx)
            weights = self.get_weights(idx)
            self.owp[idx] = np.nansum(safe_divide(opp_wins, opp_played) * weights)

    def update_oowp(self, home_idx, away_idx):
        home_oowp = np.nansum(self.owp * self.get_weights(home_idx))
        away_oowp = np.nansum(self.owp * self.get_weights(away_idx))
        self.oowp[home_idx] = home_oowp
        self.oowp[away_idx] = away_oowp

    def update_opponent_stats(self):
        pass

    def get_weights(self, idx):
        return safe_divide(self.played[idx, :], self.total_played[idx])

    def get_opp_wins(self, idx):
        return self.total_won - self.wins[:, idx]

    def get_opp_played(self, idx):
        return self.total_played - self.played[:, idx]

    def get_rpi(self, idx):
        return RPIAggregator.calculate_rpi(self.wp[idx], self.owp[idx], self.oowp[idx])

    def rate_for_games(self, teams, games):
        home_rpi = np.zeros(games.shape[0])
        away_rpi = np.zeros(games.shape[0])
        cols = {name: idx for idx, name in enumerate(games.columns)}
        for k, row in enumerate(games.values):
            if k > 100:
                break
            # TODO: handle nan teams
            if np.isnan(row[cols['hteam_id']]) or np.isnan(row[cols['ateam_id']]):
                continue
            i, j = self.team_index[row[cols['hteam_id']]], self.team_index[row[cols['ateam_id']]]

            t0 = time.time()
            self.update_wp(i, j)
            t1 = time.time()
            self.update_owp(i, j)
            t2 = time.time()
            self.update_oowp(i, j)
            t3 = time.time()


            home_rpi[k] = self.get_rpi(i)
            away_rpi[k] = self.get_rpi(j)


            # print awp, away_owp, away_oowp
            # print home_rpi, away_rpi

            self.update_wins(i, j, row[cols['home_outcome']], row[cols['neutral']])
            self.update_played(i, j, row[cols['neutral']])


            print t1 - t0, t2 - t1, t3 - t2

        return home_rpi, away_rpi

    def update_played(self, home_idx, away_idx, is_neutral=False):
        self.weighted_total_played[home_idx] += RPIAggregator.win_factor(True, is_neutral)
        self.weighted_total_played[away_idx] += RPIAggregator.win_factor(False, is_neutral)
        self.total_played[home_idx] += 1
        self.total_played[away_idx] += 1
        self.played[home_idx, away_idx] += 1.
        self.played[away_idx, home_idx] += 1.

    def update_wins(self, home_idx, away_idx, home_outcome, neutral=False):
        win_factor = RPIAggregator.win_factor(home_outcome, neutral)
        if home_outcome:
            self.weighted_total_won[home_idx] += win_factor
            self.total_won[home_idx] += 1
            self.wins[home_idx, away_idx] += 1.
        else:
            self.weighted_total_won[away_idx] += win_factor
            self.total_won[away_idx] += 1
            self.wins[away_idx, home_idx] += 1.


    @staticmethod
    def calculate_rpi(wp, owp, oowp):
        return 0.25 * wp + 0.5 * owp + 0.25 * oowp

    @staticmethod
    def win_factor(is_home, is_neutral=False):
        if is_neutral:
            return 1.
        elif is_home:
            return 0.6
        else:
            return 1.4

def update_owp(idx, total_won, total_played, wins, played):
    opp_wins = total_won - wins[:, idx]
    opp_played = total_played - played[:, idx]
    weights = safe_divide(played[idx, :], total_played[idx])
    update = np.nansum(safe_divide(opp_wins, opp_played) * weights)
    return update

def win_percentage(wins, played):
    weighted_total_won = np.sum(wins, axis=1)
    weighted_total_played = np.sum(played, axis=1)
    wp = weighted_total_won / weighted_total_played
    wp[np.where(~np.isfinite(wp))] = 0.
    return wp


def _vec_to_mat(vec, nrepeat=1):
    return np.dot(np.ones(nrepeat)[:, np.newaxis], vec[:, np.newaxis].T)


def top_level_owp(wins, played):
    total_played = _vec_to_mat(np.sum(played, axis=1))
    total_won = _vec_to_mat(np.sum(wins, axis=1))
    owp = (total_won - wins.T) / (total_played - played.T)
    owp[np.where(~np.isfinite(owp))] = 0.
    return owp


def arbitrary_owp(wp, played):
    """
    :param wp: vector of win percentages of length n_teams
    :param played: n_teams by n_teams matrix of games played
    :return:
    """
    weights = played.T / np.sum(played.T, axis=0)[:, np.newaxis]
    weights[np.where(~np.isfinite(weights))] = 0.
    owp = np.nansum(wp * weights, axis=1)
    return owp


def calculate_rpi(wp, owp, oowp):
    return 0.25 * wp + 0.5 * owp + 0.25 * oowp

def get_rpi(played, wplayed, wins, wwins):
    wp = win_percentage(wwins, wplayed)
    unweighted_owp = top_level_owp(wins, played)
    owp = arbitrary_owp(unweighted_owp, played)
    oowp = arbitrary_owp(owp, played)
    return calculate_rpi(wp, owp, oowp)


def get_games(season=None):
    games = pd.read_sql("SELECT * FROM games_test", DB.conn)
    # TODO: build this directly into the postgres query
    if season is not None:
        games['season'] = games.dt.map(lambda d: get_season(d))
        games = games[games.season == season]
        games = games.sort('dt')
        games.reset_index(inplace=True)
    return games
def get_season(dt):
    return dt.year if dt.month <= 6 else dt.year + 1
def get_teams(games):
    hteams = games[['hteam_id']]
    ateams = games[['ateam_id']]
    hteams = hteams.rename(columns={'hteam_id': 'team_id'})
    ateams = ateams.rename(columns={'ateam_id': 'team_id'})
    teams = pd.concat([hteams, ateams], axis=0)
    teams.drop_duplicates(inplace=True)
    teams.dropna(inplace=True)
    return teams


if __name__ == "__main__":
    games = get_games(2015)
    teams = get_teams(games)
    # data = [[3, 64, 1, 57, True, False], [3, 82, 0, 68, True, False], [2, 71, 3, 72, False, False],
    #         [1, 69, 3, 62, True, False], [0, 81, 2, 70, True, False], [2, 52, 1, 62, False, False]]
    # games = pd.DataFrame(data, columns=['hteam_id', 'home_score', 'ateam_id', 'away_score', 'home_outcome', 'neutral'])
    # teams = np.unique(games.hteam_id.values.tolist() + games.ateam_id.values.tolist())
    # teams = pd.DataFrame(teams[:, np.newaxis], columns=['team_id'])
    # teams['idx'] = range(teams.shape[0])
    # played, wplayed, wins, wwins, games = rate(teams, games)
    # rpi = get_rpi(played, wplayed, wins, wwins)
    # wp = win_percentage(wins, played)
    # unweighted_owp = top_level_owp(wins, played)
    # owp = arbitrary_owp(unweighted_owp, played)
    # oowp = arbitrary_owp(owp, played)
    # print wp[:10]
    # print owp[:10]
    # print rpi[:10]
    # wp, owp, oowp = rate2(teams, games)
    rpi = RPIAggregator(teams)
    home_rpi, away_rpi = rpi.rate_for_games(teams, games)
    games['home_rpi'] = home_rpi
    games['away_rpi'] = away_rpi
    # print wp[:10]
    # print owp[:10]
