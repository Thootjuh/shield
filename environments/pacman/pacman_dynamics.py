import numpy as np
from collections import defaultdict
ACTION_TRANSLATOR = [
    (0, 1), # Action 0: Go North
     (1,0), # Action 2: Go East
     (0,-1), # Action 3: Go South
     (-1,0),  # Action 4: Go West
]

GOAL_REWARD = 10
EATEN_REWARD = -10

class pacmanSimplified:
    def __init__(self, lag_chance):
        self.lag_chance = lag_chance
        self.width = 9
        self.height = 9
        self.nb_states = ((self.width)*(self.height))**2
        self.nb_actions = len(ACTION_TRANSLATOR)
        self.walls = [
            (1,1),
            (1,2),
            (1,3),
            (1,5),
            (1,6),
            (1,7),
            (2,3),
            (2,5),
            (3,0),
            (3,1),
            (3,3),
            (3,4),
            (3,5),
            (3,7),
            (5,0),
            (5,1),
            (5,3),
            (5,4),
            (5,5),
            (5,7),
            (5,8),
            (6,1),
            (6,3),
            (7,1),
            (7,5),
            (7,6),
            (7,7),
            (7,8),
            (8,3)
        ] 
        
        self.init = np.array([[0,0],[self.width-1, self.height-1]])
        self.goal = [self.width-1, self.height-1]
        
        self._state = np.zeros((2,2), dtype=int)
        self.reset()
       
    def reset(self):
        # reset state of player
        self._state[0][0] = self.init[0][0]
        self._state[0][1] = self.init[0][1]
        
        # Set state of ghost
        self._state[1][0] = self.init[1][0] 
        self._state[1][1] = self.init[1][1]
        
        
    def encode_int(self, x, y, ax, ay):
        if not (0 <= x < self.width and 0 <= y < self.height and 0 <= ax < self.width and 0 <= ay < self.height):
            print(f"x = {x}, y = {y}, gx = {ax}, gy = {ay}")
            raise ValueError("Input values out of range")
        return (x * self.height * self.width * self.height) + (y * self.width * self.height) + (ax * self.height) + ay

    def decode_int(self, state_int):
        ay = state_int % self.height
        state_int //= self.height

        ax = state_int % self.width
        state_int //= self.width

        y = state_int % self.height
        state_int //= self.height

        x = state_int % self.width

        return x, y, ax, ay

    def state_to_int(self):
        return self.encode_int(self._state[0][0], self._state[0][1], self._state[1][0], self._state[1][1])
    
    def set_random_state(self):
        
        #pick random state
        random_state = np.random.randint(self.nb_states)
        x, y, gx, gy = self.decode_int(random_state)
        while self._is_terminal_state(x, y, gx, gy):
            random_state = np.random.randint(self.nb_states)
            x, y, gx, gy = self.decode_int(random_state)
        
        
        # set position of player
        self._state[0][0] = x
        self._state[0][1] = y
        
        # Set state of ghost 1
        self._state[1][0] = gx
        self._state[1][1] = gy

    def _is_terminal_state(self, x,y,gx,gy):
        if x == self.goal[0] and y == self.goal[1]:
            return True
        if x==gx and y==gy:
            return True
        if (x,y) in self.walls:
            return True
        return False
    
    def is_done(self):
        x = self._state[0][0]
        y = self._state[0][1]
        
        # Reached Goal
        if x == self.goal[0] and y == self.goal[1]:
            return True
        
        # Eaten
        if x == self._state[1][0] and y == self._state[1][1]:
            return True
        
        return False
    
    def get_reward(self):
        if self._state[0][0] == self.goal[0] and self._state[0][1] == self.goal[1]:
            return GOAL_REWARD
        if self._state[0][0] == self._state[1][0] and self._state[0][1] == self._state[1][1]:
            return EATEN_REWARD
        return 0
    
    def get_reward_from_int(self, int):
        x, y, gx, gy = self.decode_int(int)
        if x == self.goal[0] and y == self.goal[1]:
            return GOAL_REWARD
        if x == gx and y == gy:
            return EATEN_REWARD
        return 0
    
    def is_valid_move(self, x, y, action):
        # Get the movement offset from the ACTION_TRANSLATOR
        dx, dy = ACTION_TRANSLATOR[action]

        # Compute the new position
        x_hat = x + dx
        y_hat = y + dy

        # Check if the new position is valid
        if x_hat < 0 or x_hat >= self.width or y_hat < 0 or y_hat >= self.height:
            return False # Invalid move (out of bounds)
            
        if (x_hat, y_hat) in self.walls:
            return False  # Invalid move (collision with a wall)
        
        return True  # Valid move

    def step(self, action_choice):
        # Save the old state
        old_state = self.state_to_int()
        
        #Move the player
        x = self._state[0][0]
        y = self._state[0][1]
        x_hat = 0
        y_hat = 0
        # Check if the given action choice is valid
        random_number = random_number = np.random.uniform(0.0,1.0)
        if random_number < self.lag_chance:
            x_hat = x
            y_hat = y
        else:
            dx, dy = ACTION_TRANSLATOR[action_choice]
            x_hat = max(min(x + dx, self.width-1),0)
            y_hat = max(min(y + dy, self.height-1),0)
        if (x_hat,y_hat) in self.walls:
            x_hat = self.init[0][0]
            y_hat = self.init[0][1]
        
        self._state[0][0] = x_hat
        self._state[0][1] = y_hat
        
        # Move the ghost:
        ghost_x = self._state[1][0]
        ghost_y = self._state[1][1]
        ghost_action = np.random.randint(0, 4)
        valid_action = self.is_valid_move(ghost_x, ghost_y, ghost_action)
        while not valid_action:
            ghost_action = np.random.randint(0, 4)
            valid_action = self.is_valid_move(ghost_x, ghost_y, ghost_action)
        ghost_dx, ghost_dy = ACTION_TRANSLATOR[ghost_action]
        ghost_x_hat = ghost_x + ghost_dx
        ghost_y_hat = ghost_y + ghost_dy
        self._state[1][0] = ghost_x_hat
        self._state[1][1] = ghost_y_hat
        return old_state, self.state_to_int(), self.get_reward()
    
    def get_reward_function(self):
        reward_dict = {}

        for next_state in range(self.nb_states):
            reward = self.get_reward_from_int(next_state)
            if reward != 0:
                for state in range(self.nb_states):
                    # reward_matrix[state, next_state] = self.get_reward_from_int(next_state)
                    reward_dict[(state, next_state)] = reward

        return reward_dict
    
    def in_wall(self, x,y,gx,gy):
        if (x,y) in self.walls or (gx,gy) in self.walls:
            return True
        return False
    
    def get_transition_function(self):
        transition_dict = defaultdict(float)
        counter = 0
        for state in range(self.nb_states):
            x,y,gx,gy = self.decode_int(state)
            # check if done
            if self._is_terminal_state(x,y,gx,gy) or self.in_wall(x,y,gx,gy):
                counter += 1
            else:
                for action in range(self.nb_actions):
                    # next position player
                    next_x = max(min(x + ACTION_TRANSLATOR[action][0], self.width-1),0)
                    next_y = max(min(y + ACTION_TRANSLATOR[action][1], self.height-1),0)
                    if(next_x,next_y) in self.walls:
                        next_x = self.init[0][0]
                        next_y = self.init[0][1]
                    # Possible next positions ghost
                    possible_g_actions = []
                    for g_action in range(self.nb_actions):
                        if self.is_valid_move(gx, gy, g_action):
                            next_gx = gx + ACTION_TRANSLATOR[g_action][0]
                            next_gy = gy + ACTION_TRANSLATOR[g_action][1]
                            possible_g_actions.append((next_gx, next_gy))
                    
                    for next_g_state in possible_g_actions:
                        #Player action succeeds
                        next_state = self.encode_int(next_x, next_y, next_g_state[0], next_g_state[1])
                        transition_dict[(state, action, next_state)] += (1-self.lag_chance) * (1/len(possible_g_actions))
                        #Player action fails
                        if self.lag_chance > 0:
                            next_state = self.encode_int(x, y, next_g_state[0], next_g_state[1])
                            transition_dict[(state, action, next_state)] += self.lag_chance * (1/len(possible_g_actions))

        return transition_dict