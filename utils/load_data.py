import pandas as pd
import numpy as np
from utils.rl_utils import reconfigure_index_2
def load_data(year):
    data = pd.read_csv('data/FR.csv', index_col = 0)
    data_reindexed = reconfigure_index_2(data)
    data = data_reindexed[data_reindexed.index.year == year]
    hourly_prices = data['Price'].values   
    n_hours = len(hourly_prices)
    n_days = n_hours // 24
    assert n_hours % 24 == 0, 'Some samples are missing'
    prices_array = np.array(hourly_prices).reshape(n_days, 24)
    preds = np.load(f'data/prob_forecasts_{year}.npz')
    preds_mean = preds['mean']
    preds_std = preds['std']

    return prices_array, preds_mean, preds_std