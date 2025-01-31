import numpy as np

class GridworldBaselinePolicy:
    def __init__(self, env, epsilon=0.1):
        self.nb_states = env.nb_states
        self.nb_actions = 4
        self.pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.env = env
        self.epsilon = epsilon
        
        self.compute_baseline()
        
    def compute_baseline(self):
        pi = np.zeros((self.nb_states, self.nb_actions))
        for state in range(len(pi)):
            x, y = self.env.get_state_from_int(state)
            goal_x, goal_y = self.env.goal
            
            # Determine relative position to the goal
            move_north = y > goal_y
            move_south = y < goal_y
            move_west = x > goal_x
            move_east = x < goal_x
            
            if move_north and move_west:
                pi[state][0] = 0.5  # North
                pi[state][3] = 0.5  # West
            elif move_north and move_east:
                pi[state][0] = 0.5  # North
                pi[state][1] = 0.5  # East
            elif move_south and move_west:
                pi[state][2] = 0.5  # South
                pi[state][3] = 0.5  # West
            elif move_south and move_east:
                pi[state][2] = 0.5  # South
                pi[state][1] = 0.5  # East
            elif move_north:
                pi[state][0] = 1.0  # North
            elif move_south:
                pi[state][2] = 1.0  # South
            elif move_west:
                pi[state][3] = 1.0  # West
            elif move_east:
                pi[state][1] = 1.0  # East
            
        # Apply epsilon smoothing
        self.pi = (1 - self.epsilon) * pi + (self.epsilon / self.nb_actions)
            