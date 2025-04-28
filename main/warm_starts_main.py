from utils.rl_utils import reconfigure_index, create_dict, add_noise, log_training_results
from experiments.PSO_warm_start.DQN_ import train_DQN
from experiments.PSO_warm_start.SAC_ import train_SAC
from experiments.PSO_warm_start.DDPG_ import train_DDPG
import pandas as pd

data_foler = 'data'
results_folder = 'Experiments'

print('Loading training data...')
df_train = pd.read_csv(f'{data_foler}/FR.csv')
df_train = df_train.drop(axis=1)
print(df_train.head())
print('Training data info:')
print(df_train.info())

print('Loading val data...')


