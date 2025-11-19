import gymnasium as gym
from gymnasium import Wrapper
import numpy as np

class MountainCarCrashWrapper(Wrapper):
    def __init__(self, env, crash_penalty=-250.0):
        super().__init__(env)
        self.crash_penalty = crash_penalty
        self.success_reward = 100
        self.crash_bound = -1.0
        self.max_speed = 0.1
    
    def set_state(self, state):
        """Set the internal state of the MountainCar environment."""
        base_env = self.env.unwrapped 
        assert len(state) == 2, "MountainCar state must be [position, velocity]"
        base_env.state = np.array(state, dtype=np.float32)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        position, velocity = obs
        if terminated:
            reward = self.success_reward  
            info["crash"] = False 
            print("reached Finish")
        else:
            crashed = (position <= self.crash_bound) or abs(velocity) > (0.07)
            if crashed:
                terminated = True
                reward = self.crash_penalty  # big penalty for crash
                info["crash"] = True
            else:
                info["crash"] = False

        
        return obs, reward, terminated, truncated, info