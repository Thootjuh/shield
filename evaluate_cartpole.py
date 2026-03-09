# Code for evaluating a given policy:
# Modified from: https://github.com/rems75/SPIBB-DQN
import numpy as np
import time
from discretization.MRL.helper_functions import state2region
import imageio
import os

def infer_action_mrl(state, predictor, policy, dimensions):
    region = state2region(predictor, state, dimensions)
    return int(np.argmax(policy[region]))

def infer_action(state, env, policy):
    region = env.state2region(state)
    return int(np.argmax(policy[region]))

def infer_action_DQN(state, ai):
    action, _, _, _ = ai.inference(state)
    return int(action)

def evaluate_policy(env, policy, number_of_episodes, max_nb_steps_per_episode,  disc_method, noise_factor=1.0, predictor=None, dimensions=0, ai=None, env_name="cartpole", generate_gif=False, gif_name="agent.gif", render_env=None):
    """ Evaluate the baseline number_of_epochs times for number_of_steps steps.

    Args:
      number_of_steps: number of steps to simulate during each epoch
      number_of_epochs: number of epochs to simulate
      noise_factor: the noise factor additionally applied to the environment. 1 in our experiments.
    Returns:
      Prints the mean performance on each epoch. And the mean, 10% and 1% CVAR of the performance on those epochs.
    """

    episode_count = 0
    success_count = 0
    failure_count = 0
    all_rewards = []

    frames = []

    if generate_gif:
        os.makedirs("gifs", exist_ok=True)
        gif_path = os.path.join("gifs", gif_name)

    for episode in range(number_of_episodes):
        # if epoch % 10 == 0: 
        #     print("Starting epoch {}".format(epoch), flush=True)

        # choose which environment to run
        if generate_gif and episode == 0 and render_env is not None:
            current_env = render_env
        else:
            current_env = env


        current_env.reset()
        last_state = current_env.get_state()

        if generate_gif and episode == 0 and render_env is not None:
            frame = current_env.env.render()
            if frame is not None:
                frames.append(frame)

        term, start_time = False, time.time()
        rewards, current_reward, nb_steps = [], 0, 0
        while nb_steps<max_nb_steps_per_episode and not term:
            # print(f"epoch {epoch}, step {nb_steps}")
            if disc_method == 'mrl':
                action = infer_action_mrl(last_state, predictor, policy, dimensions)
                # if nb_steps == 0 and epoch == 0:
                #     print(type(action))
                #     print(action)
            elif disc_method == 'grid':
                action = infer_action(last_state, current_env, policy)
                # if nb_steps == 0 and epoch == 0:
                #     print(type(action))
                #     print(action)
            elif disc_method == 'SPIBB-DQN':
                action = infer_action_DQN(last_state, ai)
                # if nb_steps == 0 and epoch == 0:
                    # print(type(action))
                    # print(action)
            _, _, reward = current_env.step(action)
            term = current_env.is_done()
            last_state = current_env.get_state()

            if generate_gif and episode == 0 and render_env is not None:
                frame = current_env.env.render()
                if frame is not None:
                    frames.append(frame)

            current_reward += reward
            nb_steps += 1
        
        rewards.append(current_reward)
        # print("number of steps in episode = ", nb_steps, ", total episode reward = ", current_reward, ", ended with reward ", reward)
        episode_count+=1
        if env_name=="cartpole":
            # print("A")
            if nb_steps < 50: 
                failure_count+=1
            if nb_steps >= 200:
                success_count+=1
        elif env_name=="grid":
            # print("B")
            if reward<=-1: 
                failure_count+=1
            if reward > 0:
                success_count+=1
        elif env_name=="lunar_lander":
            # print("C")
            if reward==-100: 
                # print("failure")
                failure_count+=1
            if reward == 100:
                # print("succass")
                success_count+=1
        all_rewards.append(current_reward)
        current_reward, nb_steps = 0, 0
        # print("after the episode, the total reward is now:", np.mean(all_rewards))

        if generate_gif and episode == 0 and len(frames) > 0:
            imageio.mimsave(gif_path, frames, fps=30)

    # print("success count = ", success_count)
    # print("failure count = ", failure_count)
    # print("episode count = ", episode_count)
    return np.mean(all_rewards), success_count/episode_count, failure_count/episode_count

