# Code for evaluating a given policy:
# Modified from: https://github.com/rems75/SPIBB-DQN
import numpy as np
import time



def infer_action(state, env, policy):
    region = env.state2region(state)
    return np.argmax(policy[region])

def evaluate_policy(env, policy, number_of_steps, number_of_epochs, noise_factor=1.0):
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
        if epoch % 10 == 0: 
            print("Starting epoch {}".format(epoch), flush=True)
        
        env.reset()
        last_state = env.get_state()
        term, start_time = False, time.time()
        rewards, all_nb_steps, current_reward, nb_steps, total_nb_steps = [], [], 0, 0, 0
        while total_nb_steps < number_of_steps:
            if not term:
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
        # print("Average reward: {}. Average steps: {}".format(
        #     np.mean(rewards), np.mean(all_nb_steps)), flush=True)
        # print("Epoch finished in {:.2f} seconds.\n".format(time.time() - start_time), flush=True)

    all_rewards.sort()
    print("Mean Average: {}.".format(np.mean(all_rewards)), flush=True)
    if number_of_epochs > 10:
        print("Average decile: {}.".format(np.mean(all_rewards[:int(number_of_epochs/10)])), flush=True)
    if number_of_epochs > 100:
        print("Average centile: {}".format(
            np.mean(all_rewards[:int(number_of_epochs/100)])), flush=True)
    return np.mean(all_rewards)