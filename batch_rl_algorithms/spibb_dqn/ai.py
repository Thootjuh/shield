import numpy as np
from batch_rl_algorithms.spibb_dqn.models import SmallDenseNetwork, DenseNetwork, Network, LargeNetwork, NatureNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import time
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

# Upper bound on q-values. Just used as an artefact
MAX_Q = 100000


class AI:
    def __init__(self, baseline, env, dataset, state_shape=[4], nb_actions=2, gamma=.95,
                 learning_rate=0.00025, epsilon=0.05, final_epsilon=0.05, test_epsilon=0.0, annealing_steps=1000,
                 minibatch_size=32, replay_max_size=100, update_freq=50, learning_frequency=1, ddqn=False,
                 network_size='dense', normalize=1., device=None, kappa=0.003, minimum_count=0, epsilon_soft=0):

        self.state_shape = state_shape
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.start_learning_rate = learning_rate
        self.epsilon = epsilon
        self.start_epsilon = epsilon
        self.test_epsilon = test_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = annealing_steps
        self.minibatch_size = minibatch_size
        self.network_size = network_size
        self.update_freq = update_freq
        self.update_counter = 0
        self.normalize = normalize
        self.learning_frequency = learning_frequency
        self.replay_max_size = replay_max_size

        self.ddqn = ddqn
        self.device = device
        self.network = self._build_network()
        self.target_network = self._build_network()
        self.weight_transfer(from_model=self.network, to_model=self.target_network)
        self.network.to(self.device)
        self.target_network.to(self.device)
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate, alpha=0.95, eps=1e-07)

        # SPIBB parameters
        self.baseline = baseline
        self.env = env
        self.kappa = kappa
        self.minimum_count = minimum_count
        self.epsilon_soft = epsilon_soft
        
        self.data = self.format_dataset(dataset)

    def _build_network(self):
        if self.network_size == 'small':
            return Network()
        elif self.network_size == 'large':
            return LargeNetwork(state_shape=self.state_shape, nb_channels=4, nb_actions=self.nb_actions, device=self.device)
        elif self.network_size == 'nature':
            return NatureNetwork(state_shape=self.state_shape, nb_channels=4, nb_actions=self.nb_actions, device=self.device)
        elif self.network_size == 'dense':
            return DenseNetwork(state_shape=self.state_shape[0], nb_actions=self.nb_actions, device=self.device)
        elif self.network_size == 'small_dense':
            return SmallDenseNetwork(state_shape=self.state_shape[0], nb_actions=self.nb_actions, device=self.device)
        else:
            raise ValueError('Invalid network_size.')

    def train_on_batch(self, s, a, r, s2, t):
        s = torch.FloatTensor(s).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        t = torch.FloatTensor(np.float32(t)).to(self.device)

        # Squeeze dimensions for history_len = 1
        s = torch.squeeze(s)
        s2 = torch.squeeze(s2)
        q = self.network(s / self.normalize)
        q2 = self.target_network(s2 / self.normalize).detach()
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1)
        if self.ddqn:
            q2_net = self.network(s2 / self.normalize).detach()
            q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            q2_max = torch.max(q2, 1)[0]
        bellman_target = r + self.gamma * q2_max * (1 - t)

        errs = (bellman_target - q_pred).unsqueeze(1)
        quad = torch.min(torch.abs(errs), 1)[0]
        lin = torch.abs(errs) - quad
        loss = torch.sum(0.5 * quad.pow(2) + lin)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _train_on_batch(self, s, a, r, s2, t, c, pi_b):

        s = torch.FloatTensor(s).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        t = torch.FloatTensor(np.float32(t)).to(self.device)

        # Squeeze dimensions for history_len = 1
        s = torch.squeeze(s)
        s2 = torch.squeeze(s2)

        q = self.network(s / self.normalize)
        q2 = self.target_network(s2 / self.normalize).detach()
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1)

        def _get_q2max(mask=None):
            if mask is None:
                mask = torch.FloatTensor(np.ones(c.shape)).to(self.device)
            if self.ddqn:
                q2_net = self.network(s2 / self.normalize).detach()
                a_max = torch.max(q2_net - (1-mask)*MAX_Q, 1)[1].unsqueeze(1)
                return q2.gather(1, a_max).squeeze(1), a_max
            else:
                return torch.max(q2 - (1-mask)*MAX_Q, 1)


        def _get_bellman_target_pi_b(c, pi_b):
            # All state/action counts for state s2
            c = torch.FloatTensor(c).to(self.device)
            # Policy on state s2 (estimated using softmax on the q-values)
            pi_b = torch.FloatTensor(pi_b).to(self.device)
            # Mask for "bootstrapped actions"
            mask = (c >= self.minimum_count).float()
            # r + (1 - t) * gamma * max_{a s.t. (s',a) not in B}(Q'(s',a)) * proba(actions not in B)
            #   + (1 - t) * gamma * sum(proba(a') Q'(s',a'))
            q2_max, _ = _get_q2max(mask)
            return r + (1 - t) * self.gamma * \
                (q2_max * torch.sum(pi_b*mask, 1) + torch.sum(q2 * pi_b * (1-mask), 1))

        # pi_b
        bellman_target = _get_bellman_target_pi_b(c, pi_b)

        # Huber loss
        errs = (bellman_target - q_pred).unsqueeze(1)
        quad = torch.min(torch.abs(errs), 1)[0]
        lin = torch.abs(errs) - quad
        loss = torch.sum(0.5 * quad.pow(2) + lin)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def get_q(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        return self.network(state / self.normalize).detach().cpu().numpy()

    def get_max_action(self, states, counts=[]):
        
        states = np.expand_dims(states, 0)
        q_values = self.get_q(states)[0][0]
        if self.minimum_count > 0.0:
            mask = (counts < self.minimum_count)
            region = self.env.state2region(states[0])
            policy = self.baseline[region]
            
            pi_b = np.multiply(mask, policy)
            pi_b[np.argmax(q_values - mask*MAX_Q)] += np.maximum(0, 1 - np.sum(pi_b))
            pi_b /= np.sum(pi_b)
            return np.random.choice(self.nb_actions, size=1, replace=True, p=pi_b)
        else:
            return [np.argmax(q_values)]
    def get_q_values(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        return self.network(state / self.normalize).detach().cpu().numpy()
    
    def softmax(self, x, temperature, axis=0):
        """Compute softmax values for each sets of scores in x."""
        if temperature > 0:
            e_x = np.exp((x - np.max(x)) / temperature)
            return e_x / e_x.sum(axis=axis)
        else:
            e_x = np.zeros(x.shape)
            e_x[np.argmax(x)] = 1.0
            return e_x
        
    def inference(self, state):
        q_values = self.get_q_values(state)
        # Use soft q-values
        p = self.softmax(q_values[0], temperature=0.2, axis=0)
        choice = np.random.choice(self.nb_actions, 1, p=p)[0]
        return choice, q_values, p, choice == np.argmax(p)

    def get_action(self, states, evaluate, counts=[]):
        # get action WITH exploration
        eps = self.epsilon if not evaluate else self.test_epsilon
        if np.random.binomial(1, eps):
            return np.random.randint(self.nb_actions)
        else:
            return self.get_max_action(states, counts=counts)[0]
        
    def sample(self, batch_size=1):
        s = np.zeros([batch_size] + list(self.state_shape),
                    dtype='float32')
        s2 = np.zeros([batch_size] + list(self.state_shape),
                    dtype='float32')
        t = np.zeros(batch_size, dtype='bool')
        a = np.zeros(batch_size, dtype='int32')
        r = np.zeros(batch_size, dtype='float32')
        c = np.zeros([batch_size, self.nb_actions], dtype='float32')
        p = np.zeros([batch_size, self.nb_actions], dtype='float32')
        for i in range(batch_size):
            j = np.random.randint(len(self.data['s']))
            s[i], a[i], r[i] = self.data['s'][j], self.data['a'][j], self.data['r'][j]
            s2[i] = self.data['s2'][j]
            t[i] = self.data['t'][j]
            c[i] = self.data['c'][j]
            p[i] = self.data['p'][j]
        return s, a, r, s2, t, c, p
    
    # def learn(self):
    #     """ Learning from one minibatch """
    #     assert self.minibatch_size <= self.transitions.size, 'not enough data in the pool'
    #     s, a, r, s2, term = self.transitions.sample(self.minibatch_size)
    #     self.train_on_batch(s, a, r, s2, term)
    #     if self.update_counter == self.update_freq:
    #         self.weight_transfer(from_model=self.network, to_model=self.target_network)
    #         self.update_counter = 0
    #     else:
    #         self.update_counter += 1

    def learn_on_batch(self, batch):
        objective = self._train_on_batch(*batch)
        # updating target network
        if self.update_counter == self.update_freq:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)
            self.update_counter = 0
        else:
            self.update_counter += 1
        return objective

    def anneal_eps(self, step):
        if self.epsilon > self.final_epsilon:
            decay = (self.start_epsilon - self.final_epsilon) * step / self.decay_steps
            self.epsilon = self.start_epsilon - decay
        if step >= self.decay_steps:
            self.epsilon = self.final_epsilon

    def update_lr(self, epoch):
        self.learning_rate = self.start_learning_rate / (epoch + 2)
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate

    def update_eps(self, epoch):
        self.epsilon = self.start_epsilon / (epoch + 2)

    def dump_network(self, weights_file_path):
        torch.save(self.network.state_dict(), weights_file_path)

    def load_weights(self, weights_file_path, target=False):
        self.network.load_state_dict(torch.load(weights_file_path))
        if target:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)

    def distance(self, x1, x2):
        return np.linalg.norm(x1-x2)
    
    def similarite(self, x1, x2, param, weights=None):
        return max(0, 1 - self.distance(x1, x2) / param)
    
    def compute_counts(self, states, actions, nb_actions, param=0.2, max_cdist=10000, k=200):
        """
        Computes pseudocounts for state-action pairs.
        Uses full pairwise distances for small datasets,
        and kNN approximation for large ones.
        """
        N = len(states)
        flat_states = states.reshape(N, -1)
        counts = np.zeros((N, nb_actions), dtype=np.float32)

        if N <= max_cdist:
            # ---- Small dataset: exact full pairwise distances ----
            dist_matrix = cdist(flat_states, flat_states, metric="euclidean")
            sim_matrix = np.maximum(0, 1 - dist_matrix / param)

            for i in range(N):
                np.add.at(counts[i], actions, sim_matrix[i])

        else:
            # ---- Large dataset: approximate kNN ----
            nn = NearestNeighbors(n_neighbors=min(k, N), metric="euclidean").fit(flat_states)
            distances, indices = nn.kneighbors(flat_states)

            for i in range(N):
                sims = np.maximum(0, 1 - distances[i] / param)
                for idx, s in zip(indices[i], sims):
                    counts[i, actions[idx]] += s

        return counts
    
    def format_dataset(self, dataset, param = 0.2, episodic=True):
        # dataset = [state, action_choice, next_state, reward, is_done]
        print("Computing counts. The dataset contains {} transitions.".format(len(dataset)), flush=True)

        data = {}
        data['s'] = np.zeros([len(dataset) - 1] +
                            self.state_shape, dtype='float32')
        data['s2'] = np.zeros([len(dataset) - 1] +
                            self.state_shape, dtype='float32')
        data['a'] = np.zeros((len(dataset) - 1), dtype='int32')
        data['r'] = np.zeros((len(dataset) - 1), dtype='float32')
        data['t'] = np.zeros((len(dataset) - 1), dtype='bool')
        data['c'] = np.zeros((len(dataset) - 1, self.nb_actions), dtype='float32')
        data['p'] = np.zeros((len(dataset) - 1, self.nb_actions), dtype='float32')
        
        if len(dataset) < 10000:
            for i in range(len(dataset) - 1):
                if i % 1000 == 999:
                    print('{} samples processed'.format(i))
                data['s'][i] = dataset[i][0]
                data['a'][i] = dataset[i][1]
                data['s2'][i] = dataset[i][2]
                data['r'][i] = dataset[i][3]
                data['t'][i] = dataset[i][4]
                data['p'][i] = self.baseline[self.env.state2region(dataset[i][0])]
                for j in range(len(dataset)-1):
                    s = self.similarite(dataset[i+1][0], dataset[j][0], param)
                    data['c'][i, dataset[j][1]] += s
        else: # For larger datasets (>10000), we instead use knn
            for i in range(len(dataset) - 1):
                data['s'][i] = dataset[i][0]
                data['a'][i] = dataset[i][1]
                data['s2'][i] = dataset[i][2]
                data['r'][i] = dataset[i][3]
                data['t'][i] = dataset[i][4]
                data['p'][i] = self.baseline[self.env.state2region(dataset[i][0])]
            print("computing counts")
            data['c'] = self.compute_counts(
                states=data['s'],
                actions=data['a'],
                nb_actions=self.nb_actions,
                param=param
            )
        return data
    
    @staticmethod
    def weight_transfer(from_model, to_model):
        to_model.load_state_dict(from_model.state_dict())

    def __getstate__(self):
        _dict = {k: v for k, v in self.__dict__.items()}
        del _dict['device']  # is not picklable
        del _dict['transitions']  # huge object (if you need the replay buffer, save its contnts with np.save)
        return _dict