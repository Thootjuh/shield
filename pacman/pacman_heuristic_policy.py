import numpy as np

class PacmanBaselinePolicy:
    def __init__(self, env, epsilon=0.1):
        self.nb_states = env.nb_states
        self.nb_actions = env.nb_actions
        self.pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.env = env
        self.epsilon = epsilon
        
        self.compute_baseline()
        
    def compute_baseline(self):
        pi = np.zeros((self.nb_states, self.nb_actions))
        for state in range(len(pi)):
            x, y, gx, gy = self.env.decode_int(state)

            # Define the preferred actions
            preferred_actions = [1, 0]  # Right (1), Up (0)
            fallback_actions = [2, 3]   # South (2), West (3)

            # Check which preferred actions are valid
            valid_preferred = [a for a in preferred_actions if self.env.is_valid_move(x, y, a)]

            if valid_preferred:
                # If both are valid, split probability equally
                for a in valid_preferred:
                    pi[state, a] = 1 / len(valid_preferred)
            else:
                # Check fallback actions
                valid_fallback = [a for a in fallback_actions if self.env.is_valid_move(x, y, a)]
                
                if valid_fallback:
                    for a in valid_fallback:
                        pi[state, a] = 1 / len(valid_fallback)

        self.pi = (1 - self.epsilon) * pi + self.epsilon * self.pi

            

                
            
            
            