import csv
import numpy as np
import os
import time
from batch_rl_algorithms.spibb_dqn.ai import AI

class spibb_dqn:
    def __init__(self, baseline, gamma, dataset=None, env=None, episode_max_len=1000, folder_name='/experiments',
               minimum_count=0, max_start_nullops=0):

        self.dataset = dataset
        self.last_episode_steps = 0
        self.score_agent = 0
        self.env = env
        self.ai = AI(baseline, env, dataset, state_shape=[4], nb_actions=2, gamma=gamma, minibatch_size=64, minimum_count=minimum_count)
        self.folder_name = folder_name
        self.max_start_nullops = max_start_nullops
        self.episode_max_len = episode_max_len
    
    def learn(self, number_of_epochs=1, steps_per_test=10000, exp_id=0, passes_on_dataset=100, **kwargs):

        # filename = os.path.join(self.folder_name, "spibb_{}_{}.csv".format(exp_id, self.ai.minimum_count))
        target_updates=50000
        total_steps, updates = 0, 0
        minibatch_size = self.ai.minibatch_size

        # How many updates per full pass over the dataset
        updates_per_epoch = len(self.dataset) // minibatch_size

        # How many passes we need to hit ~target_updates
        passes_on_dataset = max(1, target_updates // updates_per_epoch)
        for epoch in range(number_of_epochs):
            begin = time.time()
            print('=' * 30, flush=True)
            print('>>>>> Epoch  ' + str(epoch) + '/' + str(number_of_epochs - 1) + '  >>>>>', flush=True)
            for pass_on_dataset in range(passes_on_dataset):
                if pass_on_dataset % 10 == 9:
                    print('>>>>> Pass  ' + str(pass_on_dataset) + '/' + str(passes_on_dataset - 1) + '  >>>>>', flush=True)
                steps = 0
                while steps < len(self.dataset):
                    self.ai.learn_on_batch(self.ai.sample(self.ai.minibatch_size))
                    steps += self.ai.minibatch_size
                    total_steps += self.ai.minibatch_size
                    # Update learning rate every pass on the dataset or every 20000 steps whichever is larger
                    if 0 <= total_steps % max(round(len(self.dataset)/5), len(self.dataset)) < self.ai.minibatch_size:
                        self.ai.update_lr(updates)
                        updates += 1

        print('>>>>> Training ran in {} seconds.'.format(time.time() - begin), flush=True)
    
    
    def evaluate_policy(self, number_of_steps, number_of_epochs):
        """ Evaluate the baseline number_of_epochs times for number_of_steps steps.

        Args:
        number_of_steps: number of steps to simulate during each epoch
        number_of_epochs: number of epochs to simulate
        noise_factor: the noise factor additionally applied to the environment. 1 in our experiments.
        Returns:
        Prints the mean performance on each epoch. And the mean, 10% and 1% CVAR of the performance on those epochs.
        """

        all_rewards = []
        for epoch in range(number_of_epochs):
            # if epoch % 10 == 0: 
            #     print("Starting epoch {}".format(epoch), flush=True)
            
            self.env.reset()
            last_state = self.env.get_state()
            term, start_time = False, time.time()
            rewards, all_nb_steps, current_reward, nb_steps, total_nb_steps = [], [], 0, 0, 0
            while total_nb_steps < number_of_steps:
                if not term:
                    action, _, _, _ = self.ai.inference(last_state)
                    _, _, reward = self.env.step(action)
                    term = self.env.is_done()
                    last_state = self.env.get_state()
                    current_reward += reward
                    nb_steps += 1
                else:
                    self.env.reset()
                    last_state = self.env.get_state()
                    rewards.append(current_reward)
                    all_nb_steps.append(nb_steps)
                    total_nb_steps += nb_steps
                    current_reward, nb_steps = 0, 0
                    term = False

            all_rewards.append(np.mean(rewards))
        return np.mean(all_rewards)


        
        
        