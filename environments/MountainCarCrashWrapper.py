import gymnasium as gym
from gymnasium import Wrapper

class MountainCarCrashWrapper(Wrapper):
    def __init__(self, env, crash_penalty=-250.0):
        super().__init__(env)
        self.crash_penalty = crash_penalty
        self.success_reward = 100
        self.crash_bound = -1.0
        self.max_speed = 0.1

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        position, velocity = obs
        if terminated:
            reward = self.success_reward  
            info["crash"] = False 
        else:
            crashed = (position <= self.crash_bound) or abs(velocity) > (0.07)
            if crashed:
                terminated = True
                reward = self.crash_penalty  # big penalty for crash
                info["crash"] = True
            else:
                info["crash"] = False

        
        return obs, reward, terminated, truncated, info