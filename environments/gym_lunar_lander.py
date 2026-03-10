import gymnasium as gym
import IPython.display as ipd
import numpy as np
import environments
import random
import discretization.grid.partition as prt
from collections import defaultdict
from collections.abc import Mapping

class RewardDict(Mapping):
    def __init__(self, nb_states, terminal_idx, goal_idx, reward_per_next):
        self.nb_states = nb_states
        self.terminal_idx = terminal_idx
        self.goal_idx = goal_idx
        self.reward_per_next = reward_per_next
            
    def __getitem__(self, key):
        state, next_state = key

        if next_state == self.goal_idx:
            return 100
        if next_state == self.terminal_idx:
            return -100

        return self.reward_per_next.get(next_state, 0.0)

    def __iter__(self):
        raise RuntimeError("Full iteration over N^2 reward table is infeasible.")

    def __len__(self):
        return self.nb_states * self.nb_states
class LunarLander:
    
    def __init__(self, seed, render_mode=False):
        if render_mode:
            env = gym.make("CustomLander-v0", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="rgb_array")
        else:
            env = gym.make("CustomLander-v0", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)
        self.seed = seed
        self.env = env.env.env
        observation, _ = self.env.reset(seed=seed)
        self.init = observation
        self.state = observation
        self.state_shape = [8]
        self.terminated = False
        
        # self.goal = [0.0, 0.0, 0.0, 0.0]
        self.partition_states()
        
        self.traps = [self.nb_states - 1]
        self.goal = []



    def reset(self):
        # The self.env.reset function also resets the overall environment, which is not what we want. 
        # We probably have to rewrite the code to additionally have a reset_position function
        observation, _ = self.env.reset(seed=self.seed)
        self.state = observation
    
    
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

        #TODO dont hard code these values, get them from the env
        W = 600 / 30.0
        H = 400 / 30.0
        helipad_y = self.env.helipad_y
        LEG_DOWN = 18
        SCALE = 30.0

        for _ in range(max_attempts):

            # ----- Sample position first -----
            x = np.random.uniform(-0.95, 0.95)  # enforce |x| < 1 strictly
            y = np.random.uniform(0.1, 1.5)    # reasonable vertical band

            angle = np.random.uniform(-0.4, 0.4)  # restrict tilt

            vx = np.random.uniform(-2.0, 2.0)
            vy = np.random.uniform(-2.0, 2.0)
            omega = np.random.uniform(-2.0, 2.0)

            # Convert normalized → world pose
            pos_x = x * (W / 2) + (W / 2)
            pos_y = y * (H / 2) + (helipad_y + LEG_DOWN / SCALE)

            position = (pos_x, pos_y)

            # ---- Reject crash configurations ----
            if self.env.lander_body_overlaps_moon(position, angle):
                continue

            # ---- Check leg contact geometry ----
            legs_touch = self.env.lander_legs_overlap_moon(position, angle)

            # ---- Enforce consistent configuration ----
            if legs_touch:
                # Stable landing configuration required
                if abs(angle) > 0.2:
                    continue
                if abs(vx) > 1.0 or abs(vy) > 1.0:
                    continue
                if abs(omega) > 1.0:
                    continue

                left_leg = 1.0
                right_leg = 1.0
            else:
                left_leg = 0.0
                right_leg = 0.0

            # Construct normalized state
            state = np.array([
                x,
                y,
                vx,
                vy,
                angle,
                omega,
                left_leg,
                right_leg
            ], dtype=np.float32)

            # Final safety check
            if abs(state[0]) >= 1.0:
                continue

            self.env.set_state(state)
            # print(f"We selected state: {state}")
            return state

        raise RuntimeError("Failed to sample strictly physically plausible state.")
    def partition_states(self):
        # Non-terminal region definitions for LunarLander
        # (coarse discretization example; adjust as needed)
        nrPerDim = [8, 8, 4, 4, 8, 2, 2, 2]
        # nrPerDim = [8, 8, 6, 6, 8, 2, 2, 2]
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
        
        # Add one extra region for trap states and one for goal states
        terminal_region_idx = len(partition["center"])
        partition["terminal_idx"] = terminal_region_idx
        
        terminal_region_idx = len(partition["center"])+1
        partition["goal_idx"] = terminal_region_idx
        
        self.partition = partition
        self.nb_states = len(partition["center"]) + 2
        
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
        if self.state2region(next_state) is None:
            terminated = True
            reward = -100
            self.terminated = terminated and not truncated
            self.done = terminated or truncated
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
    
    
    def get_reward_for_cell(self, cell):

        # if self.cell_contains_goal(cell):
        #     return 100
        
        # if self.cell_contains_crash(cell):
        #     return -100
                
        if cell == self.get_traps():
            print("set reward for trap")
            return -100
        
        if cell == self.get_goal_state():
            print("set reward for goal")
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

    
    def get_reward_function(self):
        reward_per_next = {}
        # self.traps = [self.nb_states - 1]
        # self.goal = []
        
        for next_state in range(self.nb_states):
            reward = self.get_reward_for_cell(next_state)
            if reward != 0:
                reward_per_next[next_state] = reward
                # if reward == -100:
                #     self.traps.append(next_state)
                # elif reward == 100:
                #     self.goal.append(next_state)

        return RewardDict(
            nb_states=self.nb_states,
            terminal_idx=self.partition["terminal_idx"],
            goal_idx=self.partition["goal_idx"],
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
    
    # def get_traps(self):
    #     return self.traps
    
    def get_traps(self):
        return self.partition["terminal_idx"]
            
    # def get_goal_state(self):
        # return self.goal
    
    def get_goal_state(self):
        return self.partition["goal_idx"]
    
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
    
    def heuristic(self, s):
        """
        The heuristic for
        1. Testing
        2. Demonstration rollout.

        Args:
            env: The environment
            s (list): The state. Attributes:
                s[0] is the horizontal coordinate
                s[1] is the vertical coordinate
                s[2] is the horizontal speed
                s[3] is the vertical speed
                s[4] is the angle
                s[5] is the angular speed
                s[6] 1 if first leg has contact, else 0
                s[7] 1 if second leg has contact, else 0

        Returns:
            a: The heuristic to be fed into the step function defined above to determine the next step and reward.
        """

        angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
        if angle_targ > 0.4:
            angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4:
            angle_targ = -0.4
        hover_targ = 0.55 * np.abs(
            s[0]
        )  # target y should be proportional to horizontal offset

        angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
        hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

        if s[6] or s[7]:  # legs have contact
            angle_todo = 0
            hover_todo = (
                -(s[3]) * 0.5
            )  # override to reduce fall speed, that's all we need after contact

        if self.env.env.unwrapped.continuous:
            a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
            a = np.clip(a, -1, +1)
        else:
            a = 0
            if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
                a = 2
            elif angle_todo < -0.05:
                a = 3
            elif angle_todo > +0.05:
                a = 1
        return a
    
    def compute_baseline(self):
        pi_h = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        for state in range(self.nb_states-2):
            center = self.env.partition["center"][state]
            action = self.heuristic(center)
            pi_h[state][:] = 0.0
            pi_h[state][action] = 1.0

        # pi_h[-1][:] = 1/len(pi_h[0])
        self.pi = (1 - self.epsilon) * pi_h + self.epsilon * self.pi
        
        np.savetxt("real_baseline.txt", self.pi)
                
    def compute_baseline_size(self, nb_states, data):
        pi_r = np.ones((nb_states, self.nb_actions)) / self.nb_actions
        
        pi_selected = np.zeros((nb_states, self.nb_actions))
        for traj in data:
            for trans in traj:
                state = trans[1]
                action = trans[0]
                pi_selected[state][action]+=1
        
        best_actions = np.argmax(pi_selected, axis=1)
        pi_deterministic = np.zeros_like(pi_selected, dtype=float)
        pi_deterministic[np.arange(pi_selected.shape[0]), best_actions] = 1.0
        
        pi_b = (1 - self.epsilon) * pi_deterministic + self.epsilon * pi_r
        return pi_b
    
    
    
    
    
    