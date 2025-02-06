import numpy as np

ACTION_TRANSLATOR = [
    (0, 1), # Action 0: Go North
     (1,0), # Action 2: Go East
     (0,-1), # Action 3: Go South
     (-1,0),  # Action 4: Go West
]

TRAPS = [ # 10 x 10 grid
    (0,4),
    (1,4),
    (3,4),
    (4,4),
    (5,4),
    (8,4),
    (9,4),
    (4,0),
    (4,1),
    (4,3),
    (4,5),
    (4,8),
    (4,9)   
]

# TRAPS = [ #7 x 7 grid
#     (0,0),
#     (0,3),
#     (2,3),
#     (3,0),
#     (3,2),
#     (3,3),
#     (3,6),
#     (6,3)
# ]

# TRAPS = [  #5 x 5
#     (0,2),
#     (2,2),
#     (2,0)
# ]

REWARD = 10
NEGATIVE_REWARD = -1


class gridWorld:
    def __init__(self, height, width, slip_p, escape_p):
        self.height = height -1
        self.width = width -1
        self.slip_p = slip_p
        self.escape_p = escape_p
        self.nb_states = width * height
        self.nb_actions = len(ACTION_TRANSLATOR)
        self._state = np.zeros(2, dtype=int)
        self.init = np.array([0, self.height])
        self.goal = np.array([self.width, 0])
        self.traps = self.get_traps()
        self.reset()
        
    
    def get_traps(self):
        traps=[]
        for trap in TRAPS:
            traps.append(self.get_int_from_state(trap))
        return traps
    
    def get_state_int(self):
        return self.get_int_from_state(self._state)
    
    def get_int_from_state(self, state):
        return int(state[0] * (self.width+1) + state[1])
    
    def get_state_from_int(self, number):
        x = number // (self.width+1)
        y = number % (self.width+1)
        
        return x, y
    
    def get_reward(self):
        if self._state[0]==self.goal[0] and self._state[1]==self.goal[1]: 
            return REWARD
        return 0
    
    def is_finished(self):
        if self._state[0] == self.goal[0] and self._state[1]==self.goal[1]:
            return True
        return False
    
    def reset(self):
        self._state[0] = self.init[0]
        self._state[1] = self.init[1]
    
    def step(self, action):
        x = self._state[0]
        y = self._state[1]
        old_state = self.get_state_int()

        # Free
        if self.get_int_from_state([x,y]) not in self.traps:
            action_coordinates = ACTION_TRANSLATOR[action]
            random_number = np.random.uniform(0.0,1.0)
            if random_number < 1-4*self.slip_p: # Do intended action
                x_hat = min(max(x+action_coordinates[0], 0), self.width)
                y_hat = min(max(y+action_coordinates[1], 0), self.height)
            elif random_number < 1-3*self.slip_p: # slip north
                x_hat = min(max(x, 0), self.width)
                y_hat = min(max(y+1, 0), self.height)
            elif random_number < 1-2*self.slip_p: # slip East
                x_hat = min(max(x+1, 0), self.width)
                y_hat = min(max(y, 0), self.height)
            elif random_number < 1-self.slip_p: # slip South
                x_hat = min(max(x, 0), self.width)
                y_hat = min(max(y-1, 0), self.height)
            else: # Slip west
                x_hat = min(max(x-1, 0), self.width)
                y_hat = min(max(y, 0), self.height)                
                
        # Trapped
        else:
        # Here, we read choices 0 and 1 as attempting the 'Free' action and choices 2 and 3 as the 'Reset' action
            if action in [0,1]: # Free action
                random_number = np.random.uniform(0.0,1.0)
                if random_number < 1-4*self.escape_p:
                    x_hat = x
                    y_hat = y
                elif random_number < 1-3*self.escape_p: # Escape north
                    x_hat = min(max(x, 0), self.width)
                    y_hat = min(max(y+1, 0), self.height)
                elif random_number < 1-2*self.escape_p: # Escape East
                    x_hat = min(max(x+1, 0), self.width)
                    y_hat = min(max(y, 0), self.height)
                elif random_number < 1-self.escape_p: # Escape South
                    x_hat = min(max(x, 0), self.width)
                    y_hat = min(max(y-1, 0), self.height)
                else: # Escape west
                    x_hat = min(max(x-1, 0), self.width)
                    y_hat = min(max(y, 0), self.height)      
            else: # Do the reset action
                x_hat = self.init[0]
                y_hat = self.init[1]
        self._state[0] = x_hat
        self._state[1] = y_hat
        new_state = self.get_state_int()
        # if new_state > 21:
        #     print("new state", new_state)
        return old_state, new_state, self.get_reward()
    
    def get_reward_function(self):
        reward_matrix = np.zeros((self.nb_states, self.nb_states))
        goal_state = self.get_int_from_state(self.goal)
        reward_matrix[:, goal_state] = REWARD
        # print(reward_matrix)
        return reward_matrix
    
    def get_transition_function(self):
        transition_function = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        init = self.get_int_from_state(self.init)
        for state in range(len(transition_function)):
            if state not in self.traps: #If we are in an non-trapped state
                for action in range(len(transition_function[state])): 
                    # Add mass for performing action
                    x,y = self.get_state_from_int(state)
                    action_coordinates = ACTION_TRANSLATOR[action]
                    next_x = min(max(x+action_coordinates[0], 0), self.width)
                    next_y = min(max(y+action_coordinates[1], 0), self.height)
                    next_state = self.get_int_from_state([next_x, next_y])
                    transition_function[state][action][next_state] += 1-4*self.slip_p
                    
                    # Add mass for slipping
                    for transition in ACTION_TRANSLATOR:
                        next_x = min(max(x+transition[0], 0), self.width)
                        next_y = min(max(y+transition[1], 0), self.height)
                        next_state = self.get_int_from_state([next_x, next_y])
                        transition_function[state][action][next_state] += self.slip_p
            else:
                # Action Free
                # Add mass for failing to escape
                x,y = self.get_state_from_int(state)
                next_x = x
                next_y = y
                next_state = self.get_int_from_state([next_x, next_y])
                transition_function[state][0][next_state] += 1-(4*self.escape_p)
                transition_function[state][1][next_state] += 1-(4*self.escape_p)
                # Add mass for escaping
                for transition in ACTION_TRANSLATOR:
                    next_x = min(max(x+transition[0], 0), self.width)
                    next_y = min(max(y+transition[1], 0), self.height)
                    next_state = self.get_int_from_state([next_x, next_y])
                    transition_function[state][0][next_state] += self.escape_p
                    transition_function[state][1][next_state] += self.escape_p
                        
                # Action Reset
                
                transition_function[state][2][init] = 1
                transition_function[state][3][init] = 1
        goal = self.get_int_from_state(self.goal)
        transition_function[goal][:][:] = 0
        transition_function[goal][:][:] = 0
        
        # for state in range(len(transition_function)):
        #     print(f"in state {self.get_state_from_int(state)}, we have the following transition functions:")
        #     for action in range(len(transition_function[state])):
        #          print(f"for action {action}: {transition_function[state][action]}")
        return transition_function
                
                        
                    
                    
    
    
        
                
            
            
    
    