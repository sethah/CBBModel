import numpy as np
from datetime import datetime, timedelta
import psycopg2
import pandas as pd

conn = psycopg2.connect(database="cbb", user="seth", password="abc123",
                        host="localhost", port="5432")
cur = conn.cursor()

class AdjustedStat(object):
    home_factor = 1.014
    available_stats = {'efg', 'trt', 'ftr', 'ppp'}

    def __init__(self, stat, dt):
        self.dt = dt
        self.date_seq = datetime.strftime(dt, '%Y%m%d')
        assert stat in AdjustedStat.available_stats, "available stats are: %s" % AdjustedStat.available_stats
        self.stat = stat
        self.dayzero = self.start_date()
        self.daynum = (self.dt - self.dayzero).days

    def start_date(self):
        q = """SELECT dayzero FROM seasons WHERE season=%s""" % self.season()
        cur.execute(q)
        return cur.fetchone()[0]

    def date_string(self):
        return datetime.strftime(self.dt, '%Y-%m-%d')

    def season(self):
        return self.dt.year if self.dt.month < 6 else self.dt.year + 1

    def team_index(self):
        """
        INPUT: DataSummarizer, GROUPBY, STRING
        OUTPUT: DATAFRAME, Int, Int, Float

        This method creates a row index mapping for each team in the form
        of {team_id: row_index}
        """
        q = """SELECT team_id FROM teams"""
        cur.execute(q)
        results = cur.fetchall()

        self.team_indices = {results[k][0]: k for k in xrange(len(results))}
        self.nteams = len(self.team_indices)

    def preseason_rank(self):
        """
        INPUT: AdjustedStat
        OUTPUT: 2D Numpy Array, 2D Numpy Array

        This method generates a preseason offensive and defensive rank
        matrices. For ppp we initialize to Ken Pomeroy's preseason rankings.
        For others, initialize to constants.
        """
        preseason_o = np.zeros((self.nteams, 1))
        preseason_d = np.zeros((self.nteams, 1))

        cols = {'ncaaid': 0, 'adjoe': 1, 'adjde': 2, 'team_id': 3}
        q = """ SELECT a.ncaaid, a.adjoe, a.adjde, b.team_id
                FROM kenpom a
                JOIN teams b
                ON a.ncaaid=b.ncaaid
                AND a.year={season}
            """.format(season=self.season()-1)
        cur.execute(q)
        for team in cur.fetchall():
            adjo = team[cols['adjoe']] / float(100)
            adjd = team[cols['adjde']] / float(100)
            ncaaid = team[cols['ncaaid']]
            teamid = team[cols['team_id']]

            # TODO: I don't have preseason numbers for these adjusted stats
            # TODO: and these are blatant guesses
            if self.stat == 'trt':
                adjo = 0.2
                adjd = 0.2
            elif self.stat == 'efg':
                adjo = 0.35
                adjd = 0.35
            elif self.stat == 'ftr':
                adjo = 0.25
                adjd = 0.25
            preseason_d[self.team_indices[teamid]] = adjd
            preseason_o[self.team_indices[teamid]] = adjo

        avg_o = np.mean(preseason_o[preseason_o > 0])
        avg_d = np.mean(preseason_d[preseason_d > 0])

        # for teams who were not found, initialize to average of all teams
        preseason_o[preseason_o==0] = avg_o
        preseason_d[preseason_d==0] = avg_d
        return preseason_o, preseason_d

    def calc_stat(self, detailed):
        detailed['wposs'] = 0.96*(detailed.wfga - detailed.wor + 0.475*detailed.wfta)
        detailed['lposs'] = 0.96*(detailed.lfga - detailed.lor + 0.475*detailed.lfta)

        detailed['wppp'] = detailed.wscore / detailed.wposs
        detailed['lppp'] = detailed.lscore / detailed.lposs

        return detailed

    def stat_query(self):
        q = {}
        q['ppp'] =  """ SELECT
                            wloc,
                            wteam,
                            lteam,
                            wscore / (0.96*(wfga - wor + wto + 0.475*wfta)) AS wppp,
                            lscore / (0.96*(lfga - lor + lto + 0.475*lfta)) AS lppp
                        FROM reg_detailed 
                        WHERE daynum + (SELECT dayzero FROM seasons WHERE season=%s) < '%s'
                        AND season=%s
                    """ % (self.season(), self.date_string(), self.season())
        q['ftr'] =  """ SELECT
                            wloc,
                            wteam,
                            lteam,
                            wfta / CAST(wfga AS REAL) AS wftr,
                            lfta / CAST(lfga AS REAL) AS lftr
                        FROM reg_detailed 
                        WHERE daynum + (SELECT dayzero FROM seasons WHERE season=%s) < '%s'
                        AND season=%s
                    """ % (self.season(), self.date_string(), self.season())
        q['trt'] =  """ SELECT
                            wloc,
                            wteam,
                            lteam,
                            wto / (0.96*(wfga - wor + wto + 0.475*wfta)) AS wtrt,
                            lto / (0.96*(lfga - lor + lto + 0.475*lfta)) AS ltrt
                        FROM reg_detailed 
                        WHERE daynum + (SELECT dayzero FROM seasons WHERE season=%s) < '%s'
                        AND season=%s
                    """ % (self.season(), self.date_string(), self.season())
        q['efg'] =  """ SELECT
                            wloc,
                            wteam,
                            lteam,
                            (wfgm + 0.5*wfgm3) / wfga AS wefg,
                            (lfgm + 0.5*lfgm3) / lfga AS lefg
                        FROM reg_detailed 
                        WHERE daynum + (SELECT dayzero FROM seasons WHERE season=%s) < '%s'
                        AND season=%s
                    """ % (self.season(), self.date_string(), self.season())

        return q[self.stat]

    def filter_dataframe(self):
        cols = {'wloc': 0, 'wteam': 1, 'lteam': 2,
                'wstat': 3, 'lstat': 4}
        q = self.stat_query()

        cur.execute(q)
        return cur.fetchall(), cols

    def initialize(self):
        self.team_index()
        # detailed = pd.read_sql("""SELECT * FROM reg_detailed""", conn)
        stats, cols = self.filter_dataframe()
        # stats = self.calc_stat(detailed)

        raw_omat = np.empty((40, self.nteams))
        raw_dmat = np.empty((40, self.nteams))
        ind_mat = np.empty((40, self.nteams))
        loc_mat = np.empty((40, self.nteams))

        raw_omat.fill(np.nan)
        raw_dmat.fill(np.nan)
        ind_mat.fill(np.nan)
        loc_mat.fill(np.nan)
        r, c = raw_omat.shape

        for idx, game in enumerate(stats):
            stat = game[cols['wstat']]
            opp_stat = game[cols['lstat']]
            team = game[cols['wteam']]
            opp = game[cols['lteam']]

            if game[cols['wloc']] == 'H':
                loc_factor = AdjustedStat.home_factor
            elif game[cols['wloc']] == 'A':
                loc_factor = 1 / AdjustedStat.home_factor
            else:
                loc_factor = 1
            
            team_idx = self.team_indices[team]
            opp_idx = self.team_indices[opp]
            last_entry = raw_omat[r - 1][team_idx]

            non_nan_o = np.count_nonzero(~np.isnan(raw_omat[:, team_idx]))
            non_nan_d = np.count_nonzero(~np.isnan(raw_dmat[:, team_idx]))
            raw_omat[non_nan_o][team_idx] = stat
            raw_dmat[non_nan_d][team_idx] = opp_stat
            ind_mat[non_nan_o][team_idx] = opp_idx
            loc_mat[non_nan_o][team_idx] = loc_factor

            stat = game[cols['lstat']]
            opp_stat = game[cols['wstat']]
            team = game[cols['lteam']]
            opp = game[cols['wteam']]

            loc_factor = 1 / loc_factor

            team_idx = self.team_indices[team]
            opp_idx = self.team_indices[opp]
            last_entry = raw_omat[r - 1][team_idx]

            non_nan_o = np.count_nonzero(~np.isnan(raw_omat[:, team_idx]))
            non_nan_d = np.count_nonzero(~np.isnan(raw_dmat[:, team_idx]))
            raw_omat[non_nan_o][team_idx] = stat
            raw_dmat[non_nan_d][team_idx] = opp_stat
            ind_mat[non_nan_o][team_idx] = opp_idx
            loc_mat[non_nan_o][team_idx] = loc_factor

        return raw_omat, raw_dmat, ind_mat, loc_mat

    def weight_vector(self, n, wtype='linear'):
        if n == 0:
            w = np.array([1])
            return w[:, np.newaxis], 0
        elif wtype == 'linear':
            w = np.array(xrange(1, n+1))
            w = w*(1/float(w.sum()))
        else:
            w = np.ones(n) / n

        w = w[:, np.newaxis]
        c = 0.4
        n_pre = 10
        w_pre = c - c*n/(n_pre)
        w_pre = max(0, w_pre)  # don't return anything less than zero
        w = w*(1./(w.sum()/(1 - w_pre)))

        return w, w_pre

    def weight_matrix(self, raw_omat, wtype=''):
        if wtype == 'linear':
            weights = np.zeros(raw_omat.shape)
            w_pre = np.ones(raw_omat.shape[1])
            for c in xrange(raw_omat.shape[1]):
                col = raw_omat[:,c]
                n = np.sum(~np.isnan(col))
                if n == 0:
                    continue
                w, wp = self.weight_vector(n, wtype='linear')
                weights[:n,c] = w.ravel()
                w_pre[c] = wp
            return weights, w_pre
        else:
            game_counts = np.sum(~np.isnan(raw_omat), axis=0)
            weights = np.ones(raw_omat.shape)
            return weights / game_counts.astype(float), w_pre

    def rank(self):
        """
        INPUT: DataSummarizer
        OUTPUT: 2D Numpy Array, 2D Numpy Array

        This method iterates to find the adjusted offensive and defensive
        ranking matrices.
        """
        raw_omat, raw_dmat, ind_mat, loc_mat = self.initialize()
        adj_d = np.nanmean(raw_dmat, axis=0)
        adj_o = np.nanmean(raw_omat, axis=0)
        avg_o = np.nanmean(raw_omat)
        avg_d = np.nanmean(raw_dmat)

        preseason_o, preseason_d = self.preseason_rank()

        weights, w_pre = self.weight_matrix(raw_omat, wtype='linear')

        for i in xrange(20):
            adj_dprev = adj_d*1
            adj_oprev = adj_o*1

            new_omat = raw_omat / adj_d[np.nan_to_num(ind_mat).astype(int)] * loc_mat * weights * avg_o
            new_dmat = raw_dmat / adj_o[np.nan_to_num(ind_mat).astype(int)] * (1 / loc_mat) * weights * avg_d

            adj_o = np.sum(np.nan_to_num(new_omat), axis=0) + w_pre*preseason_o.ravel() 
            adj_d = np.sum(np.nan_to_num(new_dmat), axis=0) + w_pre*preseason_d.ravel()
            r_off = np.linalg.norm(np.nan_to_num(adj_oprev - adj_o))
            r_def = np.linalg.norm(np.nan_to_num(adj_dprev - adj_d))

        return adj_o, adj_d

    def print_ranks(self, ranks, n=10, reverse=True):
        teams = pd.read_sql("""SELECT * FROM teams""", conn)
        rank_list = []
        for idx, row in teams.iterrows():
            rank = ranks[self.team_indices[row.team_id]]
            if rank == 0:
                continue
            rank_list.append((row.team_name, rank))

        print sorted(rank_list, key=lambda x: x[1], reverse=reverse)[:n]

    def store_ranks(self, ortg, drtg):
        cols = {'season': 0, 'daynum': 1, 'wteam': 2, 'lteam': 3,
                'w%s' % self.stat: 4, 'l%s' % self.stat: 5,
                'wd%s' % self.stat: 6, 'ld%s' % self.stat: 7}
        q = """ SELECT 
                    season,
                    daynum,
                    wteam,
                    lteam,
                    w{stat},
                    l{stat},
                    wd{stat},
                    ld{stat}
                FROM reg_advanced
                WHERE daynum + (SELECT dayzero FROM seasons WHERE season={season}) = '{dt}'
                AND season={season}
            """.format(stat=self.stat, season=self.season(), dt=self.date_string())
        cur.execute(q)
        vals = []
        results = cur.fetchall()
        if len(results) == 0:
            return None
        for game in results:
            wteam = game[cols['wteam']]
            lteam = game[cols['lteam']]
            season = game[cols['season']]
            daynum = game[cols['daynum']]
            wrtg = ortg[self.team_indices[int(wteam)]]
            lrtg = ortg[self.team_indices[int(lteam)]]
            wdrtg = drtg[self.team_indices[int(wteam)]]
            ldrtg = drtg[self.team_indices[int(lteam)]]
            vals.append((wteam, lteam, season, daynum, wrtg, lrtg, wdrtg, ldrtg))
        # print vals[:40]
        q = """ UPDATE reg_advanced AS t SET
                w{stat}=c.w{stat},
                l{stat}=c.l{stat},
                wd{stat}=c.wd{stat},
                ld{stat}=c.ld{stat},
                wteam=c.wteam,
                lteam=c.lteam,
                season=c.season,
                daynum=c.daynum
                FROM (values {vals})
                AS c(wteam, lteam, season, daynum, w{stat}, l{stat}, wd{stat}, ld{stat})
                WHERE c.wteam=t.wteam
                AND c.lteam=t.lteam
                AND c.season=t.season
                AND c.daynum=t.daynum;
            """.format(vals=','.join(['%s' % str(v) for v in vals]),
                       stat=self.stat)
        cur.execute(q)
        conn.commit()

def store_season(year, stats=['ppp']):
    start_date = datetime(year, 11, 01).date()
    end_date = datetime(year+1, 4, 10).date()
    day_count = (end_date - start_date).days + 1

    for stat in stats:
        for single_date in (start_date + timedelta(n) for n in xrange(day_count)):
            a = AdjustedStat(stat, single_date)
            adj_o, adj_d = a.rank()
            print 'storing %s.....' % stat
            a.store_ranks(adj_o, adj_d)
            print single_date, a.daynum

if __name__ == '__main__':
    store_season(2014, stats=['ppp', 'efg', 'ftr', 'trt'])
    # d = datetime(2013, 3, 14).date()
    # a = AdjustedStat('ftr', d)


""" select r.*, t.team_name, t.kenpom
    from reg_advanced r
    join teams t
    on r.lteam=t.team_id
    where wftr is not null
    """

""" SELECT
        wteam,
        lteam,
        wloc,
        CASE
            WHEN wloc='H' then wteam
            WHEN wloc='A' then lteam
            ELSE NULL
        END AS hteam,
        CASE
            WHEN wloc='H' then lteam
            WHEN wloc='A' then wteam
            ELSE NULL
        END AS ateam
    FROM reg_advanced
"""
""" SELECT a.*, c.wscore, c.lscore, c.loc
    FROM reg_advanced a
    JOIN reg_compact c
    ON a.season=c.season
    AND a.daynum=c.daynum
    AND a.wteam=c.wteam
    AND a.wftr IS NOT NULL
"""