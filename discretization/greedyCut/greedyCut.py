import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict


class GreedyCut:
    def __init__(self, state_dim, trajectories, B=20, bounds=None,
                 initial_splits=None, binary_dims=None):

        self.state_dim = state_dim
        self.trajectories = trajectories
        self.B = B

        self.binary_dims = binary_dims if binary_dims is not None else []

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

        self.lows = np.array([b[0] for b in bounds])
        self.highs = np.array([b[1] for b in bounds])
        self.ranges = self.highs - self.lows
        self.ranges[self.ranges == 0] = 1e-8

        # --------------------------------------------------
        # Handle initial splits
        # --------------------------------------------------
        if initial_splits is None:
            initial_splits = [4] * state_dim  # default

        assert len(initial_splits) == state_dim, \
            "initial_splits must match state_dim"

        # --------------------------------------------------
        # Initialize G in NORMALIZED space
        # --------------------------------------------------
        self.G = []

        for d in range(state_dim):
            if d in self.binary_dims:
                # Binary dimension → fixed
                self.G.append([0.0, 1.0])
            else:
                n_splits = initial_splits[d]
                grid = np.linspace(0.0, 1.0, n_splits + 1).tolist()
                self.G.append(grid)

        # --------------------------------------------------
        # Normalize dataset
        # --------------------------------------------------
        self.dataset = []
        for traj in trajectories:
            for (s, a, ns, r, term, trunc) in traj:
                s_norm = self.normalize(s)
                ns_norm = self.normalize(ns)
                self.dataset.append((s_norm, a, ns_norm))

        self.action_to_data = defaultdict(list)
        for s, a, ns in self.dataset:
            self.action_to_data[a].append((s, ns))

        self.f_cache = {}

    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------
    def normalize(self, state):
        state = np.array(state)
        return (state - self.lows) / self.ranges

    def denormalize(self, state):
        state = np.array(state)
        return state * self.ranges + self.lows

    # --------------------------------------------------
    # Dynamics
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
    # state → region
    # --------------------------------------------------
    def state2region(self, state, G=None, return_id=True):
        if G is None:
            G = self.G

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
            centroid_norm = []
            for d in range(self.state_dim):
                grid = G[d]
                c = 0.5 * (grid[indices[d]] + grid[indices[d] + 1])
                centroid_norm.append(c)

            return self.denormalize(np.array(centroid_norm))

        region_id = 0
        multiplier = 1

        for d in reversed(range(self.state_dim)):
            region_id += indices[d] * multiplier
            multiplier *= (len(G[d]) - 1)

        return region_id

    # --------------------------------------------------
    # f_bar
    # --------------------------------------------------
    def f_bar(self, state, action, G=None):
        next_state = self.f(state, action)

        centroid = self.state2region(
            self.denormalize(next_state), G, return_id=False
        )

        return self.normalize(centroid)

    # --------------------------------------------------
    # Trajectories
    # --------------------------------------------------
    def compute_true_trajectory(self, traj):
        states = []
        for (s, a, ns, r, term, trunc) in traj:
            states.append(self.normalize(s))
        states.append(self.normalize(traj[-1][2]))
        return states

    def compute_discrete_trajectory(self, traj, G):
        states = []

        s_bar = self.normalize(traj[0][0])
        states.append(s_bar)

        for (s, a, ns, r, term, trunc) in traj:
            s_bar = self.f_bar(s_bar, a, G)
            states.append(s_bar)

        return states

    # --------------------------------------------------
    # Cost
    # --------------------------------------------------
    def Cost(self, true_traj, disc_traj):
        return sum(np.linalg.norm(xt - xbar) ** 2
                   for xt, xbar in zip(true_traj, disc_traj))

    # --------------------------------------------------
    # Cut
    # --------------------------------------------------
    def Cut(self, d, i, G):
        new_G = deepcopy(G)
        grid = new_G[d]

        midpoint = 0.5 * (grid[i] + grid[i + 1])
        new_G[d] = grid[:i + 1] + [midpoint] + grid[i + 1:]

        return new_G

    # --------------------------------------------------
    # Greedy
    # --------------------------------------------------
    def Greedy(self):
        G = deepcopy(self.G)
        Theta = self.trajectories

        for _ in tqdm(range(self.B)):
            traj = random.choice(Theta)

            best_cost = float("inf")
            best_d, best_i = None, None

            true_traj = self.compute_true_trajectory(traj)
            disc_traj = self.compute_discrete_trajectory(traj, G)
            best_cost = self.Cost(true_traj, disc_traj)
            
            for d in range(self.state_dim):

                if d in self.binary_dims:
                    continue

                for i in range(len(G[d]) - 1):

                    G_candidate = self.Cut(d, i, G)
                    disc_traj = self.compute_discrete_trajectory(traj, G_candidate)

                    tmp_cost = self.Cost(true_traj, disc_traj)

                    if tmp_cost < best_cost:
                        best_cost = tmp_cost
                        best_d, best_i = d, i

            # Apply cut
            if best_d is not None:
                G = self.Cut(best_d, best_i, G)

        self.G = G
        return G

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    def get_discretized_dataset(self, return_id=True):
        discretized_trajectories = []

        for traj in self.trajectories:
            new_traj = []

            for (s, a, ns, r, term, trunc) in traj:
                s_region = self.state2region(s, return_id=return_id)
                ns_region = self.state2region(ns, return_id=return_id)

                new_traj.append([a, s_region, ns_region, r])

            discretized_trajectories.append(new_traj)

        return discretized_trajectories

    # --------------------------------------------------
    # Num regions
    # --------------------------------------------------
    def get_num_regions(self, G=None):
        if G is None:
            G = self.G

        total = 1
        for d in range(self.state_dim):
            total *= (len(G[d]) - 1)

        return total

    # --------------------------------------------------
    # region → center
    # --------------------------------------------------
    def region2centre(self, region_id, G=None):
        if G is None:
            G = self.G

        indices = [0] * self.state_dim
        remaining = region_id

        for d in reversed(range(self.state_dim)):
            size = len(G[d]) - 1
            indices[d] = remaining % size
            remaining //= size

        centroid_norm = []
        for d in range(self.state_dim):
            i = indices[d]
            grid = G[d]
            c = 0.5 * (grid[i] + grid[i + 1])
            centroid_norm.append(c)

        return self.denormalize(np.array(centroid_norm))