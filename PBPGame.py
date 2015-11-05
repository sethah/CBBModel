import pandas as pd
import numpy as np
import psycopg2
from tabulate import tabulate
from collections import defaultdict

CONN = psycopg2.connect(database="cbb", user="seth", password="abc123",
                        host="localhost", port="5432")
CUR = CONN.cursor()

class PBPGame(object):

    def __init__(self, gameid):
        self.gameid = gameid
        self.get_game()
        q = """SELECT * FROM pbp WHERE game_id=%s""" % self.gameid
        self.df = pd.read_sql(q, CONN)
        self.home_team = self.df[self.df.teamid==1].team.iloc[0]
        self.away_team = self.df[self.df.teamid==0].team.iloc[0]
        self.field_goals = {'LUM', 'LUMS', 'JM', 'JMS', 'TIM', 'TIMS',
                          'TPM', 'TPMS', 'DM', 'DMS'}
        self.made_field_goals = {'LUM', 'JM', 'TIM', 'TPM', 'DM'}

    def get_game(self):
        q = """SELECT * FROM games_ncaa WHERE game_id=%s""" % self.gameid
        CUR.execute(q)
        data = CUR.fetchone()
        self.home_score = data[4]
        self.away_score = data[3]
        self.neutral = data[5]
        self.numot = data[7]
        self.home_outcome = data[6]

    def summary_df(self):
        summary_cols = ['second_chance', 'timeout_pts', 'turnover_pts',
                        'and_one', 'charge', 'blocked', 'stolen', 'assisted',
                        'poss_time']

        self.pbp_stats = {col: {} for col in summary_cols}
        for col in summary_cols:
            hteam, ateam = self.summary_stat(col)
            self.pbp_stats[col]['home'] = hteam
            self.pbp_stats[col]['away'] = ateam
            # data.append([col, hteam[1], ateam[1]])

        # self.summary_df = pd.DataFrame(data, columns=['stat', self.home_team, self.away_team])

    def summary_stat(self, col):
        if col in {'and_one', 'charge', 'blocked', 'stolen'}:
            home_team = self.df[self.df.teamid == 1][col].sum()
            away_team = self.df[col].sum() - home_team
            if np.isnan(home_team):
                home_team = 0
            if np.isnan(away_team):
                away_team = 0

            return (home_team, str(int(home_team or 0))), (away_team, str(int(away_team or 0)))
        elif col in {'assisted'}:
            home_team = self.df[self.df.teamid == 1][col].sum()
            away_team = self.df[col].sum() - home_team
            cond1 = self.df.play.isin(self.made_field_goals)
            home_team = float(home_team) / \
                        self.df[cond1 & (self.df.teamid==1)].play.count()
            away_team = float(away_team) / \
                        self.df[cond1 & (self.df.teamid==0)].play.count()
            return (home_team, '%0.2f%%' % (home_team*100)), (away_team, '%0.2f%%' % (away_team*100))
        elif col in {'poss_time'}:
            home_team = self.df[self.df.teamid == 1][col].mean()
            away_team = self.df[self.df.teamid == 0][col].mean()
        elif col in {'timeout_pts', 'second_chance', 'turnover_pts'}:
            home_team = self.df[self.df.teamid == 1][col].mean()
            away_team = self.df[self.df.teamid == 0][col].mean()

        return (home_team, '%0.2f' % home_team), (away_team, '%0.2f' % away_team)

    def game_summary(self):
        self.summary_df()
        format_dict = {'int': {'second_chance',
                               'timeout_pts',
                               'turnover_pts'},
                       'float': {'poss_time'},
                       'percent': {'assisted'}}
       
        if self.home_score > self.away_score:
            wteam = self.home_team
            lteam = self.away_team
            wscore = self.home_score
            lscore = self.away_score
        else:
            wteam = self.away_team
            lteam = self.home_team
            wscore = self.away_score
            lscore = self.home_score
        line_len = 50
        print '*'*line_len  
        print '{wteam} defeats {lteam} {wscore}-{lscore}'\
               .format(wteam=wteam,
                       lteam=lteam,
                       wscore=wscore,
                       lscore=lscore)
        print '*'*line_len
        if self.neutral:
            print 'Neutral site game'
        print 'Home team: {hteam} \nAway team: {ateam}'\
                .format(hteam=self.home_team, ateam=self.away_team)
        print '-'*line_len
        print 'Stat Summary'
        print '-'*line_len

        vals = []
        for stat in self.pbp_stats:
            vals.append([stat, self.pbp_stats[stat]['home'][1],
                         self.pbp_stats[stat]['away'][1]])

        print tabulate(vals, tablefmt='orgtbl', headers=['stat', self.home_team, self.away_team])


if __name__ == '__main__':
    q = """SELECT DISTINCT(game_id) FROM pbp LIMIT 10"""
    CUR.execute(q)
    for gid in CUR.fetchall():
        pbp = PBPGame(gid[1])

    print pbp.df.head()