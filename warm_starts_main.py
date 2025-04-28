from utils.rl_utils import log_training_results
from utils.load_data import load_data
from experiments.PSO_warm_start.DQN_ import train_DQN
from experiments.PSO_warm_start.SAC_ import train_SAC
from experiments.PSO_warm_start.DDPG_ import train_DDPG
from env.modified_env import TradingEnv
from datetime import datetime

experiment_id = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
results_folder = f"./results/SAC_{experiment_id}"

prices_2019, preds_mean_2019, preds_std_2019 = load_data(year=2019)
prices_2020, preds_mean_2020, preds_std_2020 = load_data(year=2020)


# Create environments
cont_env_2019 = TradingEnv(prices_2019, preds_mean_2019, preds_std_2019)
cont_env_2020 = TradingEnv(prices_2020, preds_mean_2020, preds_std_2020)
disc_env_2019 = TradingEnv(prices_2019, preds_mean_2019, preds_std_2019, action_type='discrete')
disc_env_2020 = TradingEnv(prices_2020, preds_mean_2020, preds_std_2020, action_type='discrete')

# Train Models
SAC_model_init, SAC_time_init, SAC_opt_time_init, SAC_mean_reward_init, SAC_std_reward_init = train_SAC(experiment_id,  cont_env_2019, initialize_weights=True, search_algo='pso')
SAC_model_no_init, SAC_time_no_init, SAC_opt_time_no_init, SAC_mean_reward_no_init, SAC_std_reward_no_init = train_SAC(experiment_id, cont_env_2019)
DDPG_model_init, DDPG_time_init, DDPG_opt_time_init, DDPG_mean_reward_init, DDPG_std_reward_init = train_DDPG(experiment_id, cont_env_2019, initialize_weights=True, search_algo='pso')
DDPG_model_no_init, DDPG_time_no_init, DDPG_opt_time_no_init, DDPG_mean_reward_no_init, DDPG_std_reward_no_init = train_DDPG(experiment_id, cont_env_2019)
DQN_model_init, DQN_time_init, DQN_opt_time_init, DQN_mean_reward_init, DQN_std_reward_init = train_DQN(experiment_id, disc_env_2019, initialize_weights=True, search_algo='pso')
DQN_model_no_init, DQN_time_no_init, DQN_opt_time_no_init, DQN_mean_reward_no_init, DQN_std_reward_no_init = train_DQN(experiment_id, disc_env_2019)

log_training_results(
   SAC_time_init, SAC_opt_time_init, SAC_mean_reward_init, SAC_std_reward_init, results_folder, experiment_id, "SAC_init", 
)
log_training_results(
   SAC_time_no_init, SAC_opt_time_no_init, SAC_mean_reward_no_init, SAC_std_reward_no_init, results_folder, experiment_id,  "SAC_no_init",
)
log_training_results(
    DDPG_time_init, DDPG_opt_time_init, DDPG_mean_reward_init, DDPG_std_reward_init, results_folder, experiment_id, "DDPG_init", 
)
log_training_results(
    DDPG_time_no_init, DDPG_opt_time_no_init, DDPG_mean_reward_no_init, DDPG_std_reward_no_init, results_folder, experiment_id, "DDPG_no_init",
)
log_training_results(
    DQN_time_init, DQN_opt_time_init, DQN_mean_reward_init, DQN_std_reward_init, results_folder, experiment_id, "DQN_init"
)
log_training_results(
    DQN_time_no_init, DQN_opt_time_no_init, DQN_mean_reward_no_init, DQN_std_reward_no_init, results_folder,  experiment_id, "DQN_no_init",
)

