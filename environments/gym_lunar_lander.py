import gymnasium as gym
import IPython.display as ipd
import numpy as np
import environments
import random
import discretization.grid.partition as prt
from collections import defaultdict
from collections.abc import Mapping

class RewardDict(Mapping):
    def __init__(self, nb_states, terminal_idx, reward_per_next):
        self.nb_states = nb_states
        self.terminal_idx = terminal_idx
        self.reward_per_next = reward_per_next
        

            

    def __getitem__(self, key):
        state, next_state = key

        if next_state == self.terminal_idx:
            return -100

        return self.reward_per_next.get(next_state, 0.0)

    def __iter__(self):
        raise RuntimeError("Full iteration over N^2 reward table is infeasible.")

    def __len__(self):
        return self.nb_states * self.nb_states
class LunarLander:
    
    def __init__(self):
        env = gym.make("CustomLander-v0", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)
        self.env = env.env.env
        observation, _ = self.env.reset()
        self.init = observation
        self.state = observation
        self.state_shape = [8]
        self.terminated = False
        
        # self.goal = [0.0, 0.0, 0.0, 0.0]
        self.partition_states()
        
        self.traps = [self.nb_states-1]
        self.goal = []
        for s in range(self.nb_states-1):
            if s % 100 == 0:
                print(s)
            reward = self.get_reward_for_cell(s)
            if reward == 100:
                self.goal.append(s)
            elif reward == -100:
                self.traps.append(s)

    def reset(self):
        # The self.env.reset function also resets the overall environment, which is not what we want. 
        # We probably have to rewrite the code to additionally have a reset_position function
        self.env.set_state(self.init)
        self.state = self.init
    
    
    def set_random_state(self, max_attempts = 1000):
        """
        Set the LunarLander environment to a random *physically plausible* state.

        A state is considered plausible if:
            - The lander body does NOT intersect the moon.
            - Leg contact flags match actual geometric contact.
            - If no leg contact, lander is not intersecting terrain.

        No physics stepping is performed.

        Parameters
        ----------
        env : LunarLander
            Environment instance (must already be reset).
        max_attempts : int
            Maximum number of rejection attempts.

        Returns
        -------
        state : np.ndarray
            The sampled physically plausible state.

        Raises
        ------
        RuntimeError if no valid state found.
        """

        assert self.env.lander is not None, "Call env.reset() first."

        low = self.env.observation_space.low
        high = self.env.observation_space.high

        W = 600 / 30.0
        H = 400 / 30.0
        helipad_y = self.env.helipad_y
        LEG_DOWN = 18
        SCALE = 30.0

        for _ in range(max_attempts):

            # Sample candidate
            state = np.random.uniform(low, high)

            # Make leg contacts binary
            state[6] = 1.0 if state[6] >= 0.5 else 0.0
            state[7] = 1.0 if state[7] >= 0.5 else 0.0

            # Convert normalized observation -> world pose
            pos_x = state[0] * (W / 2) + (W / 2)
            pos_y = state[1] * (H / 2) + (helipad_y + LEG_DOWN / SCALE)
            angle = float(state[4])

            position = (pos_x, pos_y)

            # --- 1) Body must not intersect terrain ---
            if self.env.lander_body_overlaps_moon(position, angle):
                continue

            # --- 2) Check leg contact consistency ---
            legs_touch = self.env.lander_legs_overlap_moon(position, angle)

            if state[6] == 0.0 and state[7] == 0.0:
                # If no contact flags, legs must NOT touch
                if legs_touch:
                    continue
            else:
                # If any contact flag is 1, legs must touch
                if not legs_touch:
                    continue

            # Passed all geometric checks → plausible
            self.env.set_state(state)
            return state

        raise RuntimeError("Could not sample a physically plausible state.")
    def partition_states(self):
        # Non-terminal region definitions for LunarLander
        # (coarse discretization example; adjust as needed)
        nrPerDim = [8, 8, 4, 4, 4, 4, 2, 2]

        # Approximate observation bounds used for partitioning
        regionWidth = [
            2.0 / nrPerDim[0],        # x position ∈ [-1.0, 1.0]
            5.0 / nrPerDim[1],        # y position ∈ [-2.5, 2.5]
            20.0 / nrPerDim[2],       # x velocity ∈ [-10, 10]
            20.0 / nrPerDim[3],       # y velocity ∈ [-10, 10]
            15 / nrPerDim[4],  # angle ∈ [-2π, 2π]
            20.0 / nrPerDim[5],       # angular velocity ∈ [-10, 10]
            2.0 / nrPerDim[6],        # left leg contact ∈ [0,1]
            2.0 / nrPerDim[7]         # right leg contact ∈ [0,1]
        ]

        origin = [
            0.0, 
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5, 
            0.5
        ]

        partition = prt.define_partition(
            dim=8,
            nrPerDim=nrPerDim,
            regionWidth=regionWidth,
            origin=origin
        )

        self.partition = partition
        self.nb_states = len(partition["center"])
        
        # Add one extra region for terminal states
        terminal_region_idx = len(partition["center"])
        partition["terminal_idx"] = terminal_region_idx
        
        self.partition = partition
        self.nb_states = len(partition["center"]) + 1
        
    def state2region(self, state):
        idx = prt.state2region(state, self.partition, self.partition['c_tuple'])
        if idx == None:
            return self.partition["terminal_idx"]
        return idx[0]
    
    def step(self, action):
        old_state = self.state
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.terminated = terminated and not truncated
        self.truncated = truncated
        self.done = terminated or truncated
        self.state = next_state

        return old_state, next_state, reward
    
    def is_terminated(self):
        return self.terminated
    
    def is_truncated(self):
        return self.truncated
    
    def is_done(self):
        return self.terminated or self.truncated

    def cell_contains_crash(self, cell_idx, n_samples=50):
        """
        Check whether there exists a state inside a partition cell that corresponds
        to a crash configuration (lander body touching moon).

        No simulation is performed. Pure geometric collision checking.

        Parameters
        ----------
        env : LunarLander
            Environment instance (already reset at least once).
        partition : dict
            Partition object returned by define_partition().
        cell_idx : int
            Index of the cell center in partition["center"].
        n_samples : int
            Number of sampled states inside the cell.

        Returns
        -------
        bool
        """

        assert 0 <= cell_idx <self.partition["nr_regions"]

        # Ensure terrain + bodies exist
        self.env.reset()

        low = self.partition["low"][cell_idx]
        upp = self.partition["upp"][cell_idx]

        # Extract needed env constants (no globals used)
        W = self.env.VIEWPORT_W / self.env.SCALE if hasattr(self.env, "VIEWPORT_W") else 600 / 30.0
        H = self.env.VIEWPORT_H / self.env.SCALE if hasattr(self.env, "VIEWPORT_H") else 400 / 30.0

        helipad_y = self.env.helipad_y
        LEG_DOWN = 18
        SCALE = 30.0
        FPS = 50

        for _ in range(n_samples):

            # Sample state inside the cell
            state = np.random.uniform(low, upp)

            # Leg contacts must be binary
            state[6] = 1.0 if state[6] >= 0.5 else 0.0
            state[7] = 1.0 if state[7] >= 0.5 else 0.0

            # ---- Convert normalized observation → world pose ----

            pos_x = state[0] * (W / 2) + (W / 2)
            pos_y = state[1] * (H / 2) + (helipad_y + LEG_DOWN / SCALE)

            angle = float(state[4])

            # ---- Pure geometric collision check ----

            if self.env.lander_body_overlaps_moon((pos_x, pos_y), angle):
                return True

        return False
    
    def cell_contains_goal(self, cell_idx):
        """
        Check if the point (0,0) lies inside the given cell.

        Parameters
        ----------
        partition : dict
            Partition dictionary returned by define_partition.
        cell_idx : int
            Index of the cell to check in partition['center'].

        Returns
        -------
        bool
            True if (0,0) is inside the cell, False otherwise.
        """
        x_low, y_low = self.partition['low'][cell_idx][:2]
        x_upp, y_upp = self.partition['upp'][cell_idx][:2]

        # Check if 0 is within the range for both x and y
        in_x = x_low <= 0 <= x_upp
        in_y = y_low <= 0 <= y_upp

        return in_x and in_y
    
    def get_reward_for_cell_transition(self, prev_cell, cell):

        # Terminal rewards already handled
        if self.cell_contains_crash(cell):
            return -100

        if self.cell_contains_goal(cell):
            return 100

        # Out-of-bounds in x-direction
        if abs(self.partition["center"][cell][0]) >= 1.0:
            return -100

        # ---- Shaping reward at cell center ----

        center = self.partition["center"][cell]

        x = center[0]
        y = center[1]
        vx = center[2]
        vy = center[3]
        angle = center[4]
        left_leg = center[6]
        right_leg = center[7]

        shaping = (
            -100.0 * np.sqrt(x * x + y * y)
            -100.0 * np.sqrt(vx * vx + vy * vy)
            -100.0 * abs(angle)
            +10.0 * left_leg
            +10.0 * right_leg
        )

        prev_center = self.partition["center"][prev_cell]

        prev_shaping = (
            -100.0 * np.sqrt(prev_center[0]**2 + prev_center[1]**2)
            -100.0 * np.sqrt(prev_center[2]**2 + prev_center[3]**2)
            -100.0 * abs(prev_center[4])
            +10.0 * prev_center[6]
            +10.0 * prev_center[7]
        )
        reward = shaping - prev_shaping
        # ----- Engine penalty approximation -----
        # delta_vy = center[3] - prev_center[3]

        # if delta_vy > 0:
        #     main_engine_power = min(1.0, abs(delta_vy))
        # else:
        #     main_engine_power = 0.0

        # delta_ang_vel = center[5] - prev_center[5]

        # if abs(delta_ang_vel) > 0:
        #     side_engine_power = min(1.0, abs(delta_ang_vel))
        # else:
        #     side_engine_power = 0.0

        # reward -= 0.30 * main_engine_power
        # reward -= 0.03 * side_engine_power
        return reward
    
    def get_reward_for_cell(self, cell):

        # Terminal rewards already handled
        if self.cell_contains_crash(cell):
            return -100

        if self.cell_contains_goal(cell):
            return 100

        # Out-of-bounds in x-direction
        if abs(self.partition["center"][cell][0]) >= 1.0:
            return -100

        # ---- Shaping reward at cell center ----

        center = self.partition["center"][cell]

        x = center[0]
        y = center[1]
        vx = center[2]
        vy = center[3]
        angle = center[4]
        left_leg = center[6]
        right_leg = center[7]

        shaping = (
            -100.0 * np.sqrt(x * x + y * y)
            -100.0 * np.sqrt(vx * vx + vy * vy)
            -100.0 * abs(angle)
            +10.0 * left_leg
            +10.0 * right_leg
        )

        return shaping
    
    def get_reward_function_slow(self):
        reward_function = defaultdict(float)
        for next_state in range(self.nb_states-1):
            if next_state % 100 == 0:
                print(next_state)
            if self.cell_contains_crash(next_state):
                print("A")
                for state in range(self.nb_states):
                    reward_function[(state, next_state)]=-100 # -100 reward for crashing
            elif self.cell_contains_goal(next_state):
                print("B")
                for state in range(self.nb_states):
                    reward_function[(state, next_state)]=100 # +100 reward for landing
            elif abs(self.partition["center"][next_state][0]) >= 1.0:
                print("C")
                print(self.partition["center"][next_state])
                for state in range(self.nb_states):
                    reward_function[(state, next_state)]=-100 # -100 reward for out of frame
            else: # Else, the reward depends on both the current and next state
                print("D")
                for state in range(self.nb_states):
                    reward = self.get_reward_for_cell_transition(state, next_state)
                    if reward != 0:
                        reward_function[(state, next_state)]=reward
            reward_function[(next_state, self.partition["terminal_idx"])] = -100 # negative reward for transitioning to any terminal state

        return reward_function
    
    def get_reward_function(self):
        reward_per_next = {}

        for next_state in range(self.nb_states - 1):
            reward = self.get_reward_for_cell(next_state)
            if reward != 0:
                reward_per_next[next_state] = reward

        return RewardDict(
            nb_states=self.nb_states,
            terminal_idx=self.partition["terminal_idx"],
            reward_per_next=reward_per_next,
        )
    
    def get_nb_actions(self):
        return 4
    
    def get_state_shape(self):
        return self.state_shape
    
    def get_state(self):
        return self.state
    
    def get_nb_states(self):
        return self.nb_states
    
    def get_traps(self):
        return self.traps
            
    def get_goal_state(self):
        return self.goal
    
    def get_init_state(self):
        return self.init, self.state2region(self.init)
    
    
class LunarLanderPolicy:
    def __init__ (self, env, epsilon):
        self.nb_states = env.nb_states
        self.nb_actions = 4
        self.pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.env = env
        self.epsilon = epsilon
        
        self.compute_baseline()
    
    def compute_baseline(self):
        pass
    
    def compute_baseline_size(self):
        pass
    
    
    
    
    
    