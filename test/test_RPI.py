from RPI2 import RPIAggregator
import testing_utils

import pandas as pd
import numpy as np


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
    df['game_id'] = np.arange(df.shape[0])

    teams = pd.DataFrame([[team_ids[k], team_ids[k]] for k in team_ids], columns=['team_id', 'i_team'])
    return df, teams

def _unstack(stacked):
    left = stacked
    right = stacked.rename(columns={'hteam_id': 'ateam_id', 'ateam_id': 'hteam_id'})
    unstacked = pd.concat([left, right])
    unstacked.rename(columns={'hteam_id': 'team', 'ateam_id': 'opp'}, inplace=True)
    return unstacked

def test_rpi():
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