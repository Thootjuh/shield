import numpy as np

class AirplaneBaselinePolicy:
    def __init__(self, env, epsilon=0.1):
        self.nb_states = env.maxX * env.maxY * env.maxY
        self.nb_actions = 3
        self.pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.env = env
        self.epsilon = epsilon
        
        self.compute_baseline()
        
        
        
    def compute_baseline(self):
        # Heuristic policy: If you are at the same y as the adversary plane, we move. If we are at a different y, we stay
        pi = np.zeros((self.nb_states, self.nb_actions))
        for state in range(len(pi)):
            _,y,ay = self.env.decode_int(state)
            if y == ay: # at the same height as the adversary
                pi[state][0] = 0.5 # Up
                pi[state][1] = 0.5 # Down
                pi[state][2] = 0 # Stay
            else: # At a different height than the adversary
                pi[state][0] = 0 # Up
                pi[state][1] = 0 # Down
                pi[state][2] = 1 # Stay
        self.pi = (1 - self.epsilon) * pi + self.epsilon * self.pi