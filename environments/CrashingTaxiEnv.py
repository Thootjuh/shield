import gymnasium as gym
from gymnasium.envs.toy_text.taxi import TaxiEnv
from typing import Optional
import numpy as np
from gymnasium import Env, spaces, utils
import itertools

CRASH_REWARD = -50
SUCCESS_REWARD = 50
WRONG_PICKUP = -50
WRONG_DROPOFF = -50
STEP_REWARD = -1
MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]
WINDOW_SIZE = (550, 350)

class CustomTaxiEnv(TaxiEnv):
    def __init__(self, render_mode: Optional[str] = None):
        self.desc = np.asarray(MAP, dtype="c")
        self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]
        self.goal_states = [0, 85, 410, 475, 501]
        self.combinations = [(a, b) for a, b in itertools.permutations(range(4), 2)]
        # add 3 states for possible endings:
        # -1 for crash
        # -2 for succesfull dropoff
        # -3 for failed dropoff
        self.num_states = 500+3
        
        num_rows = 5
        num_columns = 5
        self.max_row = num_rows - 1
        self.max_col = num_columns - 1
        self.initial_state_distrib = np.zeros(self.num_states)
        num_actions = 6
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(self.num_states)
        }
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx:
                            self.initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            if state in self.goal_states:
                                for new_pass_idx, new_dest_idx in self.combinations:
                                    intended_state = self.encode(row, col, new_pass_idx, new_dest_idx)
                                    self.P[state][action].append((1/len(self.combinations), intended_state, 0, False))
                            else:
                                self._build_rainy_transitions(row, col, pass_idx, dest_idx, action)
                            # defaults
                            # new_row, new_col, new_pass_idx = row, col, pass_idx
                            # reward = STEP_REWARD
                            #   # default reward when there is no pickup/dropoff
                            # terminated = False
                            # taxi_loc = (row, col)

                            # if action == 0:
                            #     new_row = min(row + 1, self.max_row)
                            # elif action == 1:
                            #     new_row = max(row - 1, 0)
                            # if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                            #     new_col = min(col + 1, self.max_col)
                            # elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                            #     new_col = max(col - 1, 0)
                            # elif action == 2 and self.desc[1 + row, 2 * col + 2] == b"|":
                            #     new_col = STEP_REWARD
                            #     terminated = True
                            #     reward = CRASH_REWARD
                            # elif action == 3 and self.desc[1 + row, 2 * col] == b"|":
                            #     new_col = STEP_REWARD
                            #     terminated = True
                            #     reward = CRASH_REWARD
                            # elif action == 4:  # pickup
                            #     if pass_idx < 4 and taxi_loc == locs[pass_idx]:
                            #         new_pass_idx = 4
                            #     else:  # passenger not at location
                            #         reward = WRONG_DROPOFF
                            # elif action == 5:  # dropoff
                            #     if (taxi_loc == locs[dest_idx]) and pass_idx == 4:
                            #         new_pass_idx = dest_idx
                            #         terminated = True
                            #         reward = SUCCESS_REWARD
                            #     elif (taxi_loc in locs) and pass_idx == 4:
                            #         new_pass_idx = locs.index(taxi_loc)
                            #     else:  # dropoff at wrong location
                            #         reward = WRONG_DROPOFF
                            # new_state = self.encode(
                            #     new_row, new_col, new_pass_idx, dest_idx
                            # )
                            # self.P[state][action].append(
                            #     (1.0, new_state, reward, terminated)
                            # )
        for act in range(num_actions):
            self.P[500][act].append((0.0, 500, 0, True))
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(self.num_states)

        self.render_mode = render_mode

        # pygame utils
        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.desc.shape[1],
            WINDOW_SIZE[1] / self.desc.shape[0],
        )
        self.taxi_imgs = None
        self.taxi_orientation = 0
        self.passenger_img = None
        self.destination_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None
    
    def pick_initial_state(self):
        in_s = np.argmax(self.initial_state_distrib)
        return in_s
    
    def set_state(self, state):
        self.s = state
    
        
    def _build_rainy_transitions(self, row, col, pass_idx, dest_idx, action):
            """Computes the next action for a state (row, col, pass_idx, dest_idx) and action for `is_rainy`."""
            state = self.encode(row, col, pass_idx, dest_idx)

            taxi_loc = left_pos = right_pos = (row, col)
            new_row, new_col, new_pass_idx = row, col, pass_idx
            reward = STEP_REWARD  # default reward when there is no pickup/dropoff
            terminated = False

            moves = {
                0: ((1, 0), (0, -1), (0, 1), (-1, 0)),  # Down
                1: ((-1, 0), (0, -1), (0, 1), (1, 0)),  # Up
                2: ((0, 1), (1, 0), (-1, 0), (0,-1)),  # Right
                3: ((0, -1), (1, 0), (-1, 0), (0,1)),  # Left
            }

            # Check if movement is allowed
            if (
                action in {0, 1}
                or (action == 2 and self.desc[1 + row, 2 * col + 2] == b":")
                or (action == 3 and self.desc[1 + row, 2 * col] == b":")
            ):
                dr, dc = moves[action][0]
                new_row = max(0, min(row + dr, self.max_row))
                new_col = max(0, min(col + dc, self.max_col))

                left_pos = self._calc_new_position(row, col, moves[action][1], offset=2)
                right_pos = self._calc_new_position(row, col, moves[action][2])
                back_pos = self._calc_new_position(row, col, moves[action][3])
                intended_state = self.encode(new_row, new_col, new_pass_idx, dest_idx)
                
            elif ((action == 2 and self.desc[1 + row, 2 * col + 2] == b"|") or (action == 3 and self.desc[1 + row, 2 * col] == b"|")):
                dr, dc = moves[action][0]
                new_row = max(0, min(row + dr, self.max_row))
                new_col = max(0, min(col + dc, self.max_col))
                reward = CRASH_REWARD
                terminated = True
                
                left_pos = self._calc_new_position(row, col, moves[action][1], offset=2)
                right_pos = self._calc_new_position(row, col, moves[action][2])
                back_pos = self._calc_new_position(row, col, moves[action][3])
                intended_state = self.num_states-1
                
            elif action == 4:  # pickup
                new_pass_idx, reward = self._pickup(taxi_loc, new_pass_idx, reward)
                if reward == WRONG_PICKUP:
                    intended_state = self.num_states-3
                    terminated = True
                else:
                    intended_state = self.encode(new_row, new_col, new_pass_idx, dest_idx)
    
                
            elif action == 5:  # dropoff
                new_pass_idx, reward, terminated = self._dropoff(
                    taxi_loc, new_pass_idx, dest_idx, reward
                )
                if reward == WRONG_DROPOFF:
                    intended_state = self.num_states-3
                else:
                    intended_state = self.encode(new_row, new_col, new_pass_idx, dest_idx)
            

            if action <= 3:
                if left_pos[2] == CRASH_REWARD:
                    left_state = self.num_states-1
                    left_terminated = True
                else:
                    left_state = self.encode(left_pos[0], left_pos[1], new_pass_idx, dest_idx)
                    left_terminated = False
                if right_pos[2] == CRASH_REWARD:
                    right_state = self.num_states-1
                    right_terminated = True
                else:
                    right_state = self.encode(right_pos[0], right_pos[1], new_pass_idx, dest_idx)
                    right_terminated = False
                if back_pos[2] == CRASH_REWARD:
                    back_state = self.num_states-1
                    back_terminated = True
                else:
                    back_state = self.encode(back_pos[0], back_pos[1], new_pass_idx, dest_idx)
                    back_terminated = False

                self.P[state][action].append((0.6, intended_state, reward, terminated))
                self.P[state][action].append((0.2, left_state, left_pos[2], left_terminated))
                self.P[state][action].append((0.2, right_state, right_pos[2], right_terminated))
                # self.P[state][action].append((0.05, back_state, back_pos[2], back_terminated))
            else:
                self.P[state][action].append((1.0, intended_state, reward, terminated))
                
    def _calc_new_position(self, row, col, movement, offset=0):
        """Calculates the new position for a row and col to the movement."""
        dr, dc = movement
        new_row = max(0, min(row + dr, self.max_row))
        new_col = max(0, min(col + dc, self.max_col))
        if self.desc[1 + new_row, 2 * new_col + offset] == b":":
            return new_row, new_col, STEP_REWARD
        elif self.desc[1 + new_row, 2 * new_col + offset] == b"|":
            return -1, -1, CRASH_REWARD
        else:  # Default to current position if not traversable
            
            return row, col, STEP_REWARD
        
    def _pickup(self, taxi_loc, pass_idx, reward):
        """Computes the new location and reward for pickup action."""
        if pass_idx < 4 and taxi_loc == self.locs[pass_idx]:
            new_pass_idx = 4
            new_reward = reward
        else:  # passenger not at location
            new_pass_idx = pass_idx
            new_reward = WRONG_PICKUP

        return new_pass_idx, new_reward, 
    
    def isGoalState(self, state):
        if state == self.num_states-1:
            return False
        elif state == self.num_states-2:
            return True
        elif state == self.num_states-3:
            return False
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)
        taxi_loc = (taxi_row, taxi_col)
        if (taxi_loc == self.locs[dest_idx]) and pass_loc == dest_idx:
            return True
        else:
            return False
        
    def _dropoff(self, taxi_loc, pass_idx, dest_idx, default_reward):
        """Computes the new location and reward for return dropoff action."""
        if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4:
            new_pass_idx = dest_idx
            new_terminated = False
            new_reward = SUCCESS_REWARD
        elif (taxi_loc in self.locs) and pass_idx == 4: #drop at wrong location
            new_pass_idx = self.locs.index(taxi_loc)
            new_terminated = True
            new_reward = WRONG_DROPOFF
        else:  # dropoff at wrong location
            new_pass_idx = pass_idx
            new_terminated = True
            new_reward = WRONG_DROPOFF
            

        return new_pass_idx, new_reward, new_terminated
    
    def action_mask(self, state: int):
        """Computes an action mask for the action space using the state information."""
        mask = np.zeros(6, dtype=np.int8)
        if state == self.num_states-1 or state == self.num_states-2 or state == self.num_states-3:
            return mask
        
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)
        if taxi_row < 4:
            mask[0] = 1
        if taxi_row > 0:
            mask[1] = 1
        if ((taxi_col < 4 and self.desc[taxi_row + 1, 2 * taxi_col + 2] == b":" ) or (taxi_col < 4 and self.desc[taxi_row + 1, 2 * taxi_col + 2] == b"|" )):
            mask[2] = 1
        if ((taxi_col > 0 and self.desc[taxi_row + 1, 2 * taxi_col] == b":") or (taxi_col > 0 and self.desc[taxi_row + 1, 2 * taxi_col] == b"|")):
            mask[3] = 1
        if pass_loc < 4 and (taxi_row, taxi_col) == self.locs[pass_loc]:
            mask[4] = 1
        if pass_loc == 4 and (
            (taxi_row, taxi_col) == self.locs[dest_idx]
            or (taxi_row, taxi_col) in self.locs
        ):
            mask[5] = 1
        return mask
    
    def decode(self, i):
        if i == self.num_states-1:
            return -1, -1, -1, -1
        if i == self.num_states-2:
            return -2, -2, -2, -2
        if i == self.num_states-3:
            return -3, -3, -3, -3
        else:
            return super().decode(i)
    
    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        if taxi_row == -1 or taxi_col == -1:
            return self.num_states-1
        elif taxi_row == -2 or taxi_col == -2:
            return self.num_states-2
        elif taxi_row == -3 or taxi_col == -3:
            return self.num_states-3
        else:
            return super().encode(taxi_row, taxi_col, pass_loc, dest_idx)