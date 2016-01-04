import pandas as pd
import numpy as np
from datetime import datetime, date

import util
from DB import DB
from RPI2 import RPIAggregator
from one_bit import OneBitAggregator

def insert_column(col_name, table_name, col_type):
    q = "SELECT column_name FROM information_schema.columns WHERE table_name='%s'" % table_name
    columns = pd.read_sql(q, DB.conn)['column_name'].values
    cur = DB.conn.cursor()
    if col_name not in columns:
        q = "ALTER TABLE %s ADD COLUMN %s %s" % (table_name, col_name, col_type)
        print q
        cur.execute(q)

def insert_ratings(season_ratings, conn):
    assert len(season_ratings) == len(util.ALL_SEASONS), \
        "number of seasons provided %s did not match \
        current number of seasons %s" % (len(season_ratings), len(util.ALL_SEASONS))
    cur = conn.cursor()

    ratings = season_ratings[0]['ratings']
    for rating_col in ratings:
        insert_column(rating_col, 'ratings', 'REAL')

    for i, season_data in enumerate(season_ratings):
        game_ids = season_data['game_ids']
        ratings = season_data['ratings']
        for j in range(len(game_ids)):
            set_string = ",".join(["%s=%s" % (rating, ratings[rating][j]) for rating in ratings])
            if j % 1000 == 0:
                print j
            if np.isnan(game_ids[j]):
                continue
            q = """
                    UPDATE ratings
                    SET {set_string}
                    WHERE game_id={gid}
                """.format(set_string=set_string, gid=game_ids[j])
            cur.execute(q)
        DB.conn.commit()

def rate_all(aggregator):
    seasons = []
    for season in util.ALL_SEASONS:
        games = util.get_games(season)
        teams = util.get_teams(games)

        season_ratings_dict = aggregator.rate_for_games(teams, games)
        season_dict = {'game_ids': games.game_id.values, 'ratings': season_ratings_dict}
        seasons.append(season_dict)
        print season

    return seasons

if __name__ == "__main__":
    rating_data = rate_all(RPIAggregator())




