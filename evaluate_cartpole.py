# Code for evaluating a given policy:
# Modified from: https://github.com/rems75/SPIBB-DQN
import numpy as np
import time
from discretization.MRL.helper_functions import state2region

def infer_action_mrl(state, predictor, policy, dimensions):
    region = state2region(predictor, state, dimensions)
    return np.argmax(policy[region])

def infer_action(state, env, policy):
    region = env.state2region(state)
    return np.argmax(policy[region])

def evaluate_policy(env, policy, number_of_steps, number_of_epochs, disc_method, noise_factor=1.0, predictor=None, dimensions=0):
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
        
        env.reset()
        last_state = env.get_state()
        term, start_time = False, time.time()
        rewards, all_nb_steps, current_reward, nb_steps, total_nb_steps = [], [], 0, 0, 0
        while total_nb_steps < number_of_steps:
            if not term:
                if disc_method == 'mrl':
                    action = infer_action_mrl(last_state, predictor, policy, dimensions)
                elif disc_method == 'grid':
                    action = infer_action(last_state, env, policy)
                _, _, reward = env.step(action)
                term = env.is_done()
                last_state = env.get_state()
                current_reward += reward
                nb_steps += 1
            else:
                env.reset()
                last_state = env.get_state()
                rewards.append(current_reward)
                all_nb_steps.append(nb_steps)
                total_nb_steps += nb_steps
                current_reward, nb_steps = 0, 0
                term = False

        all_rewards.append(np.mean(rewards))
    return np.mean(all_rewards)