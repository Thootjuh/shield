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
    



class GhostAvoidingBaselinePolicy:
    def __init__(self, env, epsilon=0.1):
        self.nb_states = env.nb_states
        self.nb_actions = env.nb_actions
        self.pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.env = env
        self.epsilon = epsilon
        
        self.compute_baseline()

    def _has_chance_eaten(self, act, x, y, gx, gy):
        if act == 0 and gy-y==2:
            return True
        if act == 1 and gx-x==2:
            return True
        if act == 2 and y-gy==2:
            return True
        if act == 3 and x-gx==2:
            return True
        return False
    
    def compute_baseline(self):
        pi = np.zeros((self.nb_states, self.nb_actions))
        for state in range(len(pi)):
            x, y, gx, gy = self.env.decode_int(state)

            # Define the preferred actions
            preferred_actions = [1, 0]  # Right (1), Up (0)
            fallback_actions = [2, 3]   # South (2), West (3)

            # Check which preferred actions are valid
            valid_preferred = [a for a in preferred_actions if self.env.is_valid_move(x, y, a) and not self._has_chance_eaten(a, x,y,gx,gy)]

            if valid_preferred:
                # If both are valid, split probability equally
                for a in valid_preferred:
                    pi[state, a] = 1 / len(valid_preferred)
            else:
                # Check fallback actions
                valid_fallback = [a for a in fallback_actions if self.env.is_valid_move(x, y, a) and not self._has_chance_eaten(a, x,y,gx,gy)]
                if valid_fallback:
                    for a in valid_fallback:
                        pi[state, a] = 1 / len(valid_fallback)
            # if we have no safe actions, we pick any random valid action
            if sum(pi[state]) != 1.0:
                valid_actions = [a for a in [0,1,2,3] if self.env.is_valid_move(x, y, a)]
                for a in valid_actions:
                    pi[state, a] = 1 / len(valid_actions)

        self.pi = (1 - self.epsilon) * pi + self.epsilon * self.pi

            
            

                
            
            
            