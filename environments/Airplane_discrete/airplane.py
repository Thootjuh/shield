import numpy as np
import random

CRASH_PENALTY = -10
SUCCEED_REWARD = 5

ACTION_TRANSLATOR = [
    -1, # Action 1: Go down
     1, # Action 2: Go Up
     0, # Action 3: Stay
]

class Airplane:
    def __init__(self, maxY, maxX, R, P):
        self.maxY = maxY
        self.maxX = maxX
        self.response = R
        self.adv_prob = P
        self._state = np.zeros((3), dtype=int) #x, y, ay x is mirrored for the adversarial plane
        self.nb_states = maxX*maxY*maxY
        self.reset()
        # Set init state
        
    def reset(self):
        # Reset x to 0
        self._state[0] = 0
        
        # set both y's to the middle y
        middle = int(self.maxY/2)
        self._state[1] = middle
        self._state[2] = middle
        
    
    def encode_int(self, x, y, ay):
        if not (0 <= x < self.maxX and 0 <= y < self.maxY and 0 <= ay < self.maxY):
            print(x, y, ay)
            raise ValueError("Input values out of range")
        return (x * self.maxY * self.maxY) + (y * self.maxY) + ay
        
    def decode_int(self, state_int):   
        ay = state_int % self.maxY
        state_int //= self.maxY
        
        y = state_int % self.maxY
        state_int //= self.maxY
        
        x = state_int % self.maxX
        
        return x, y, ay
    
    def get_state_int(self):
        return self.encode_int(self._state[0], self._state[1], self._state[2])
    
    def is_done_state(self):
        return self.is_done(self._state[0])
    
    def is_done(self, x):
        return x == self.maxX-1
    
    def get_reward(self, x, y, ay):
        if x == self.maxX-1:
            if y==ay: 
                return CRASH_PENALTY
            return SUCCEED_REWARD
        return 0
    
    def get_reward_state(self):
        return self.get_reward(self._state[0], self._state[1], self._state[2])
    
    def step(self, action_choice):
        old_state = self.get_state_int()
        x,y,ay = self.decode_int(old_state)
        
        # find new x
        x_new = x+1
        if x_new >= self.maxX:
            x_new = x
            return old_state, old_state, 0
        self._state[0]=x_new
        
        # find new y
        random_number = np.random.uniform(0.0,1.0)
        if random_number <= self.response:    
            y_new = y+ACTION_TRANSLATOR[action_choice]
            if y_new >= self.maxY or y_new < 0:
                y_new = y
        else:
            y_new = y
        self._state[1]=y_new     
           
        # find new ay
        adv_random_number = np.random.uniform(0.0,1.0)
        if adv_random_number <= self.adv_prob: # Adversary goes up
            adv_y_new = ay+1
            if adv_y_new >= self.maxY or adv_y_new < 0:
               adv_y_new = ay
        elif adv_random_number > self.adv_prob and adv_random_number <= self.adv_prob*2: # Adversary goes down
            adv_y_new = ay-1
            if adv_y_new >= self.maxY or adv_y_new < 0:
               adv_y_new = ay
        else: # Adversary stays
            adv_y_new = ay
        self._state[2]= adv_y_new   
        
        new_state = self.get_state_int()
        return old_state, new_state, self.get_reward_state()
    
    def get_reward_function(self):
        reward_matrix = np.zeros((self.nb_states, self.nb_states))
        for state in range(self.nb_states):
            for next_state in range(self.nb_states):
                x,y,ay = self.decode_int(next_state)
                reward_matrix[state, next_state] = self.get_reward(x,y,ay)
        return reward_matrix
    
    def get_transition_function(self):
        nb_actions = len(ACTION_TRANSLATOR)
        transition_matrix = np.zeros((self.nb_states, nb_actions, self.nb_states))
        for state in range(len(transition_matrix)):
            x, y, ay = self.decode_int(state)
            if self.is_done(x):
                transition_matrix[state, :, state] = 0
            else:
                x_next = x+1
                for action in range(len(transition_matrix[state])):
                    y_next = max(min(y+ACTION_TRANSLATOR[action], self.maxY-1), 0)
                    
                    # adv goes down
                    ay_next = max(0, ay-1)
                    # controll action succeeds
                    next_state = self.encode_int(x_next, y_next, ay_next)
                    transition_matrix[state, action, next_state] += self.response*self.adv_prob 
                    # controll action fails 
                    next_state = self.encode_int(x_next, y, ay_next)
                    transition_matrix[state, action, next_state] += (1-self.response)*self.adv_prob 
                    
                    # adv goes up
                    ay_next = min(self.maxY-1, ay+1)
                    next_state = self.encode_int(x_next, y_next, ay_next)
                    transition_matrix[state, action, next_state] += self.response*self.adv_prob
                    # controll action fails 
                    next_state = self.encode_int(x_next, y, ay_next)
                    transition_matrix[state, action, next_state] += (1-self.response)*self.adv_prob 
                    
                    # adv stays
                    ay_next = ay
                    next_state = self.encode_int(x_next, y_next, ay_next)
                    transition_matrix[state, action, next_state] += self.response*(1-2*self.adv_prob)
                    # controll action fails 
                    next_state = self.encode_int(x_next, y, ay_next)
                    transition_matrix[state, action, next_state] += (1-self.response)*(1-2*self.adv_prob)
                
        return transition_matrix
            
