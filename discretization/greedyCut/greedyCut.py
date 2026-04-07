import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict


class GreedyCut:
    def __init__(self, state_dim, trajectories, B=20, bounds=None):
        self.state_dim = state_dim
        self.trajectories = trajectories
        self.B = B

        # --------------------------------------------------
        # Handle bounds
        # --------------------------------------------------
        if bounds is None:
            all_states = []
            for traj in trajectories:
                for (s, a, ns, r, term, trunc) in traj:
                    all_states.append(s)
                    all_states.append(ns)

            all_states = np.array(all_states)

            bounds = []
            for d in range(state_dim):
                low = np.min(all_states[:, d])
                high = np.max(all_states[:, d])
                bounds.append((low, high))

        self.bounds = bounds

        # Convert bounds to arrays
        self.lows = np.array([b[0] for b in bounds])
        self.highs = np.array([b[1] for b in bounds])
        self.ranges = self.highs - self.lows
        self.ranges[self.ranges == 0] = 1e-8  # avoid division by zero

        # --------------------------------------------------
        # Initialize G in NORMALIZED space
        # --------------------------------------------------
        self.G = [[0.0,0.25,0.5,0.75,1.0] for _ in range(state_dim)]

        # --------------------------------------------------
        # Normalize dataset for dynamics
        # --------------------------------------------------
        self.dataset = []
        for traj in trajectories:
            for (s, a, ns, r, term, trunc) in traj:
                s_norm = self.normalize(s)
                ns_norm = self.normalize(ns)
                self.dataset.append((s_norm, a, ns_norm))

        # Group by action
        self.action_to_data = defaultdict(list)
        for s, a, ns in self.dataset:
            self.action_to_data[a].append((s, ns))

        self.f_cache = {}

    # --------------------------------------------------
    # Normalization utilities
    # --------------------------------------------------
    def normalize(self, state):
        state = np.array(state)
        return (state - self.lows) / self.ranges

    def denormalize(self, state):
        state = np.array(state)
        return state * self.ranges + self.lows

    # --------------------------------------------------
    # Approximate dynamics f (in normalized space)
    # --------------------------------------------------
    def f(self, state, action):
        key = (tuple(state), action)

        if key in self.f_cache:
            return self.f_cache[key]

        best_dist = float("inf")
        best_next = state

        for s, ns in self.action_to_data[action]:
            dist = np.linalg.norm(state - s)
            if dist < best_dist:
                best_dist = dist
                best_next = ns

        self.f_cache[key] = best_next
        return best_next

    # --------------------------------------------------
    # Map state → region
    # --------------------------------------------------
    def state2region(self, state, G=None, return_id=True):
        if G is None:
            G = self.G

        # Normalize input
        state_norm = self.normalize(state)

        indices = []

        for d in range(self.state_dim):
            val = state_norm[d]
            grid = G[d]

            val = np.clip(val, grid[0], grid[-1])

            for i in range(len(grid) - 1):
                if grid[i] <= val <= grid[i + 1]:
                    indices.append(i)
                    break

        indices = tuple(indices)

        if not return_id:
            # Return centroid in ORIGINAL space
            centroid_norm = []
            for d in range(self.state_dim):
                grid = G[d]
                c = 0.5 * (grid[indices[d]] + grid[indices[d] + 1])
                centroid_norm.append(c)

            return self.denormalize(np.array(centroid_norm))

        # Convert to integer ID
        region_id = 0
        multiplier = 1

        for d in reversed(range(self.state_dim)):
            region_id += indices[d] * multiplier
            multiplier *= (len(G[d]) - 1)

        return region_id

    # --------------------------------------------------
    # Discretized dynamics f_bar (normalized space)
    # --------------------------------------------------
    def f_bar(self, state, action, G=None):
        next_state = self.f(state, action)

        # Map to region centroid (in original → then normalize again)
        centroid = self.state2region(
            self.denormalize(next_state), G, return_id=False
        )

        return self.normalize(centroid)

    # --------------------------------------------------
    # True trajectory (normalized)
    # --------------------------------------------------
    def compute_true_trajectory(self, traj):
        states = []
        for (s, a, ns, r, term, trunc) in traj:
            states.append(self.normalize(s))
        states.append(self.normalize(traj[-1][2]))
        return states

    # --------------------------------------------------
    # Discretized trajectory
    # --------------------------------------------------
    def compute_discrete_trajectory(self, traj, G):
        states = []

        s0 = self.normalize(traj[0][0])
        s_bar = s0
        states.append(s_bar)

        for (s, a, ns, r, term, trunc) in traj:
            s_bar = self.f_bar(s_bar, a, G)
            states.append(s_bar)

        return states

    # --------------------------------------------------
    # Cost function
    # --------------------------------------------------
    def Cost(self, true_traj, disc_traj):
        cost = 0.0
        for xt, xbar in zip(true_traj, disc_traj):
            cost += np.linalg.norm(xt - xbar) ** 2
        return cost

    # --------------------------------------------------
    # Cut function
    # --------------------------------------------------
    def Cut(self, d, i, G):
        new_G = deepcopy(G)
        grid = new_G[d]

        midpoint = 0.5 * (grid[i] + grid[i + 1])
        new_G[d] = grid[:i + 1] + [midpoint] + grid[i + 1:]

        return new_G

    # --------------------------------------------------
    # Greedy algorithm
    # --------------------------------------------------
    def Greedy(self):
        G = deepcopy(self.G)

        Theta = self.trajectories

        for _ in tqdm(range(self.B)):
            traj = random.choice(Theta)

            best_cost = float("inf")
            worst_cost = -float("inf")

            best_d, best_i = None, None

            true_traj = self.compute_true_trajectory(traj)

            for d in range(self.state_dim):
                for i in range(len(G[d]) - 1):

                    G_candidate = self.Cut(d, i, G)
                    disc_traj = self.compute_discrete_trajectory(traj, G_candidate)

                    tmp_cost = self.Cost(true_traj, disc_traj)

                    if tmp_cost < best_cost:
                        best_cost = tmp_cost
                        best_d, best_i = d, i

                    if tmp_cost > worst_cost:
                        worst_cost = tmp_cost

            # Tie-breaking
            if best_cost == worst_cost:
                random_state = random.choice(true_traj)
                for d in range(self.state_dim):
                    for i in range(len(G[d]) - 1):
                        if G[d][i] <= random_state[d] <= G[d][i + 1]:
                            best_d, best_i = d, i
                            break

            G = self.Cut(best_d, best_i, G)

        self.G = G
        return G

    # --------------------------------------------------
    # Discretized dataset
    # --------------------------------------------------
    def get_discretized_dataset(self, return_id=True):
        discretized_trajectories = []
        nb_states = self.get_num_regions()
        for traj in self.trajectories:
            new_traj = []

            for (s, a, ns, r, term, trunc) in traj:
                s_region = self.state2region(s, return_id=return_id)
                ns_region = self.state2region(ns, return_id=return_id)
                # if term:
                #     ns_region = nb_states+1
                new_traj.append([a, s_region, ns_region, r])

            discretized_trajectories.append(new_traj)

        return discretized_trajectories

    # --------------------------------------------------
    # Number of regions
    # --------------------------------------------------
    def get_num_regions(self, G=None):
        if G is None:
            G = self.G

        total = 1
        for d in range(self.state_dim):
            total *= (len(G[d]) - 1)

        return total