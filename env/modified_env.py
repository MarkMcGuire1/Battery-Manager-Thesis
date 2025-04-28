import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class TradingEnv(gym.Env):
    """
    Fixed version of the SACTradingEnv that correctly handles boundary conditions
    """
    def __init__(self, prices, forecasts, uncertainty, mode = 'Train', battery_capacity=3.6, max_power=3.6, eff_c=0.9, eff_d=0.8, action_type='continuous'):
        super(TradingEnv, self).__init__()
        
        self.prices = prices  # 2D array [days, hours]
        self.forecasts = forecasts  # forecasted prices
        self.standard_devs = uncertainty  # forecasted uncertainty
        self.mode = mode
        self.action_type = action_type.lower()
        
        # Battery parameters
        self.battery_capacity = battery_capacity  # MWh
        self.max_power = max_power  # MW (maximum charge/discharge rate)
        self.eff_c = eff_c  # charging efficiency
        self.eff_d = eff_d  # discharging efficiency
        
        if self.action_type == 'continuous':
            # Continuous action space from -1 (full discharge) to 1 (full charge)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        elif self.action_type == 'discrete':
            # Discrete action space with 5 actions: -1, -0.5, 0, 0.5, 1
            self.action_space = spaces.Discrete(3)    

        # State space: [SoC, current_price, next_price, uncertainty]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.0]), 
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.day = None
        self.hour = None
        self.soc = None
        self.current_date = None
    
    def _normalize_prices(self, prices):
        """Normalize prices to [-1, 1] range"""
        p_min, p_max = 0, 100  # Price bounds
        norm_prices = np.clip(prices, p_min, p_max)
        return 2 * (norm_prices - p_min) / (p_max - p_min) - 1
    
    def _norm(self, value, v_min, v_max):
        """Normalize a value to [0, 1] range"""
        return np.clip((value - v_min) / (v_max - v_min), 0, 1)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode (day)"""
        super().reset(seed=seed)
        
        if self.mode == 'Train':
            # Choose a random day
            self.day = np.random.randint(0, len(self.prices))
            self.hour = 0
            self.soc = 0.5  # Start with half-charged battery
        elif self.mode == 'Eval':
            self.day = self.eval_index # Fixed day for evaluation
            self.hour = 0  
            if self.eval_index == 0:
                self.soc = 0.5 # Start with half-charged battery  
        # Get observation for the initial state
        obs = self._get_observation()
        
        return obs, {}
    
    def step(self, action):
        """Execute action and return new state, reward, and done flag"""
        # Safety check - ensure hour is valid
        if self.hour >= 24:
            # If we accidentally went past the end, return terminal state
            return self._get_observation(), 0.0, True, False, {'profit': 0.0}
        
        if self.mode == 'Eval':
            self.day = self.eval_index
        
        # Get current price (unnormalized for reward calculation)
        day_prices = self.prices[self.day]
        price = day_prices[self.hour]

        if self.action_type == 'discrete':
            # Map discrete action to power level
            if action == 0:
                power = -self.max_power
            if action == 1:
                power = 0.0
            if action == 2:
                power = self.max_power
                
        elif self.action_type == 'continuous':
            # Process action (which is a numpy array from gym's perspective)
            if isinstance(action, np.ndarray):
                action = action[0]  # Extract scalar from array
            
        # Convert action to power level
        power = float(action) * self.max_power
        
        # Initialize reward
        reward = 0.0
        
        # Charging case
        if power > 0:
            available_capacity = (1.0 - self.soc) * self.battery_capacity
            actual_power = min(power, available_capacity)
            energy_stored = actual_power * self.eff_c
            self.soc += energy_stored / self.battery_capacity
            reward = -actual_power * price  # Buying energy incurs cost
            
        # Discharging case
        elif power < 0:
            available_energy = self.soc * self.battery_capacity
            actual_power = min(-power, available_energy)
            energy_delivered = actual_power * self.eff_d
            self.soc -= actual_power / self.battery_capacity
            reward = energy_delivered * price  # Selling energy earns revenue
            
        # Move to next hour
        self.hour += 1
        done = (self.hour >= 24)
        
        return self._get_observation(), float(reward), done, False, {'profit': reward}
    
    def _get_observation(self):
        """Helper to get current observation"""
        if self.mode == 'Train':
            day_prices = self.prices[self.day]
        elif self.mode == 'Eval':
            day_prices = self.prices[self.eval_index]
        # Safety check - ensure hour doesn't exceed bounds
        hour_idx = min(self.hour, 23)
        norm_prices = self._normalize_prices(day_prices)
        
        # For the next hour's price, wrap around to 0 if at end of day
        next_hour = (hour_idx + 1) % 24
        
        # Compute coefficient of variation safely
        next_price = float(day_prices[hour_idx])
        if self.mode == 'Train':
            std_dev = float(self.standard_devs[self.day][hour_idx])
        elif self.mode == 'Eval':
            std_dev = float(self.standard_devs[self.eval_index][hour_idx])
        coefficient_variation = std_dev / max(next_price, 0.1)  # Avoid division by zero
        
        # Make sure all values are scalars
        soc_norm = float(self._norm(self.soc, 0, 1))
        price_norm = float(norm_prices[hour_idx])
        next_price_norm = float(norm_prices[next_hour])
        coef_var_norm = float(self._norm(coefficient_variation, 0, 1))
        
        return np.array([
            soc_norm,
            price_norm,
            next_price_norm, 
            coef_var_norm
        ], dtype=np.float32)
    
    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        self._seed = seed
        np.random.seed(seed)
        random.seed(seed)
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(seed)
        if hasattr(self.observation_space, 'seed'):
            self.observation_space.seed(seed)
        return [seed]
    
    def render(self):
        """Print current state for debugging"""
        print(f"Day: {self.day}, Hour: {self.hour}, SoC: {self.soc:.3f}")
        if self.hour < 24:
            print(f"Current Price: {self.prices[self.day][self.hour]:.2f} €/MWh") 