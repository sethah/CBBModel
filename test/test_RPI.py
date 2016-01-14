from RPI2 import RPIAggregator
import testing_utils

import pandas as pd
import numpy as np


def rpi_test_data():
    # TODO: fix this to return the three dataframes that I need.
    team_ids = {'UConn': 0, 'Kansas': 1, 'Duke': 2, 'Minnesota': 3}
    data = [['UConn', 64, 'Kansas', 57, 65],
            ['UConn', 82, 'Duke', 68, 70],
            ['Minnesota', 71, 'UConn', 72, 63],
            ['Kansas',	69,	'UConn', 62, 65],
            ['Duke', 81, 'Minnesota', 70, 68],
            ['Minnesota', 52, 'Kansas', 62, 55]]
    df = pd.DataFrame(data, columns=['hteam', 'hscore', 'ateam', 'ascore', 'poss'])
    df['home_outcome'] = df.hscore > df.ascore
    df['neutral'] = False
    df['hppp'] = df.hscore / df.poss
    df['appp'] = df.ascore / df.poss
    df['hteam_id'] = df.hteam.map(lambda x: team_ids.get(x))
    df['i_hteam'] = df.hteam.map(lambda x: team_ids.get(x))
    df['ateam_id'] = df.ateam.map(lambda x: team_ids.get(x))
    df['i_ateam'] = df.ateam.map(lambda x: team_ids.get(x))
    df['game_id'] = np.arange(df.shape[0])

    teams = pd.DataFrame([[v, v, k] for k, v in team_ids.iteritems()], columns=['team_id', 'i_team', 'team'])
    return df, _stack(df), teams.sort('i_team')

def _stack(unstacked):
    left = unstacked
    right = unstacked.rename(columns={'hteam_id': 'ateam_id', 'ateam_id': 'hteam_id',
                                      'hteam': 'ateam', 'ateam': 'hteam', 'appp': 'hppp',
                                      'hppp': 'appp', 'i_hteam': 'i_ateam', 'i_ateam': 'i_hteam',
                                      'hscore': 'ascore', 'ascore': 'hscore'})
    stacked = pd.concat([left, right])
    stacked.rename(columns={'hteam_id': 'team_id', 'ateam_id': 'opp_id', 'hteam': 'team',
                            'ateam': 'opp', 'hppp': 'ppp', 'appp': 'oppp', 'i_hteam': 'i_team',
                            'i_ateam': 'i_opp', 'hscore': 'score', 'ascore': 'opp_score',
                            }, inplace=True)
    return stacked

def test_rpi():
    # TODO: changed retrieval methods so need to update here
    stacked = rpi_test_data()
    unstacked = _unstack(stacked)
    agg = RPIAggregator(team_col='team')
    agg.rate_for_games(stacked, unstacked)

    expected_wp = [0.65, 0.5882, 0.3, 0.0]
    expected_owp = [3. / 4., 2. / 3., 1. / 3., 0.3889]
    expected_oowp = [0.5139, 0.6296, 0.5694, 0.5833]
    expected_rpi = [0.666, 0.6378, 0.3840, 0.3403]

    abs_tol = 1e-3
    testing_utils.compare_array(expected_rpi, agg.calculate(), abs_tol=abs_tol)
    testing_utils.compare_array(expected_wp, agg.wp, abs_tol=abs_tol)
    testing_utils.compare_array(expected_owp, agg.owp, abs_tol=abs_tol)
    testing_utils.compare_array(expected_oowp, agg.oowp, abs_tol=abs_tol)
    return agg

if __name__ == "__main__":
    stacked = rpi_test_data()
    agg = test_rpi()