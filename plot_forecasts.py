import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.load_data import load_data

prices_2019, preds_2019, std_2019 = load_data(2019)
prices_2020, preds_2020, std_2020 = load_data(2020)
prices_2019_flat = np.array(prices_2019).flatten()
prices_2020_flat = np.array(prices_2020).flatten()
preds_2019_flat = np.array(preds_2019).flatten()
preds_2020_flat = np.array(preds_2020).flatten()
std_2019_flat = np.array(std_2019).flatten()
std_2020_flat = np.array(std_2020).flatten()

months = {
    'Jan':(0,30) ,'Feb': (31, 58), 'Mar': (59, 88), 'Apr':(89,118), 'May':(119,148), 'Jun':(149,178),
    'Jul': (179,209), 'Aug': (210, 240), 'Sep': (241, 270) , 'Oct': (271, 301), 'Nov': (302, 331), 'Dec': (332, 362)
}
Jan = months['Jan']
jan_start = Jan[0]
jan_end = Jan[1]

def plot_probabilistic_forecasts(prices, preds_mean, preds_std, year):
    """
    """
    prices = prices[jan_start:jan_end]
    preds_mean = preds_mean[jan_start:jan_end]
    preds_std = preds_std[jan_start:jan_end]
    prices = prices.flatten()
    preds_mean = preds_mean.flatten()
    preds_std = preds_std.flatten()

    data_range = pd.date_range(start='2019-01-01', periods=len(prices), freq='H')
    # Plotting the probabilistic forecasts
    plt.figure(figsize=(10, 6))
    plt.plot(data_range, preds_mean, label='Forecasted Mean', color='blue', linewidth=2)

    # Add the shaded area for the standard deviation
    plt.fill_between(data_range, preds_mean - preds_std, preds_mean + preds_std, color='blue', alpha=0.3, label='Forecasted Deviation')

    # Plot the real data on top
    plt.plot(data_range, prices, label='Real Data', color='red', linestyle='dashed', linewidth=2)

    # Add labels and title
    plt.title(f'Probabilistic Forecasts vs Real Data {year}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

plot_probabilistic_forecasts(prices_2019, preds_2019, std_2019, year=2019)
plot_probabilistic_forecasts(prices_2020, preds_2020, std_2020, year=2020)
