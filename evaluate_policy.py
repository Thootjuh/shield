# Code for evaluating a given policy:
# Modified from: https://github.com/rems75/SPIBB-DQN
import numpy as np
import time
from discretization.MRL.helper_functions import state2region
# from discretization.greedyCut.greedyCut import GreedyCut
import imageio
import os
import discretization.constructed.lunarLanderDisc as LLDisc

def infer_action_greedy_cut(state, predictor, policy, gc):
    region = gc.state2region(state, predictor)
    # print("region = ", region, " withc policy ", policy[region])
    return int(np.random.choice(policy.shape[1], p=policy[region]))

def infer_action_mrl(state, predictor, policy, dimensions):
    region = state2region(predictor, state, dimensions)
    return int(np.random.choice(policy.shape[1], p=policy[region]))

def infer_action(state, env, policy):
    region = env.state2region(state)
    return int(np.random.choice(policy.shape[1], p=policy[region]))

def infer_action_SPIBB_DQN(state, ai):
    action, _, _, _ = ai.inference(state)
    return int(action)

def infer_action_CQL_DQN(state, ai):
    if not isinstance(state, np.ndarray):
            state = np.array(state)
    return ai.get_action(state, epsilon=0.0)[0]

def infer_action_grid_CQL_DQN(state, ai, env):
    region = env.state2region(state)
    if region == env.partition["terminal_idx"]:
        # print("WHAT? You shouldn't be taking an action here")
        return ai.get_action(state, epsilon=0.0)[0]
    elif  region == env.partition["goal_idx"]:
        print("And deff not here...")
    centre = env.partition["center"][region]
    # print(type(state))
    # print(state)
    # print(type(centre))
    # print(centre)
    return ai.get_action(centre, epsilon=0.0)[0]
def infer_action_custom(state, policy):
    region = LLDisc.state2region(state)
    # print(policy[region])
    # print(int(np.random.choice(policy.shape[1], p=policy[region])))
    return int(np.random.choice(policy.shape[1], p=policy[region]))

def infer_action_heuristic(state, env):
    return env.heuristic(state)

    
def evaluate_policy(env, policy, number_of_episodes, max_nb_steps_per_episode,
                    disc_method, noise_factor=1.0, predictor=None, dimensions=0,
                    ai=None, env_name="cartpole", generate_gif=False,
                    gif_name="agent.gif", render_env=None, gamma=0.99):

    episode_count = 0
    success_count = 0
    failure_count = 0
    avoid_count = 0

    all_rewards = []
    all_discounted_rewards = []

    frames = []
    if env_name == "moving_obstacles":
        goal_counts = [0] * len(env.goal_region)
    if generate_gif:
        gif_folder = os.path.join("gifs", env_name)
        os.makedirs(gif_folder, exist_ok=True)
        gif_path = os.path.join(gif_folder, gif_name)

    for episode in range(number_of_episodes):
        if episode % 100 == 0:
            print(episode)
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

        term = False
        rewards = []
        current_reward = 0
        discounted_reward = 0
        discount = 1.0

        nb_steps = 0

        while nb_steps < max_nb_steps_per_episode and not term:

            if disc_method == 'mrl':
                action = infer_action_mrl(last_state, predictor, policy, dimensions)
            elif disc_method == 'grid':
                action = infer_action(last_state, current_env, policy)
            elif disc_method == 'SPIBB-DQN':
                action = infer_action_SPIBB_DQN(last_state, ai)
            elif disc_method == 'CQL-DQN':
                action = infer_action_CQL_DQN(last_state, ai)
            elif disc_method == 'GreedyCut':
                action = infer_action_greedy_cut(last_state, predictor, policy, ai)
            elif disc_method == 'test':
                action = infer_action_grid_CQL_DQN(last_state, ai, current_env)
            elif disc_method == 'heuristic':
                action = infer_action_heuristic(last_state, current_env)
            elif disc_method == 'custom':
                action = infer_action_custom(last_state, policy)
            _, _, reward = current_env.step(action)

            term = current_env.is_done()
            last_state = current_env.get_state()

            if generate_gif and episode == 0 and render_env is not None:
                frame = current_env.env.render()
                if frame is not None:
                    frames.append(frame)

            # standard return
            current_reward += reward

            # discounted return
            discounted_reward += discount * reward
            discount *= gamma

            nb_steps += 1

        episode_count += 1
        if nb_steps > max_nb_steps_per_episode:
            print("AAAAAAAAAAAAAAAAAAAAA")
        if env_name == "cartpole":
            if nb_steps < 50:
                failure_count += 1
            if nb_steps >= 50:
                avoid_count += 1
            if nb_steps >= 200:
                success_count += 1

        elif env_name == "grid":
            if reward <= -1:
                failure_count += 1
            if reward > -1:
                avoid_count += 1
            if reward > 0:
                success_count += 1

        elif env_name == "lunar_lander":
            if reward == -100:
                failure_count += 1
            if reward != -100:
                avoid_count += 1
            if reward == 100:
                success_count += 1
            if disc_method=='heuristic':
                if reward != -100 and reward != 100:
                    print(f"heuristic reward = {reward}")

        elif env_name == "frozen_lake_cont":
            if reward < 0:
                failure_count += 1
            if reward >= 0:
                avoid_count += 1
            if reward > 0:
                success_count += 1
        
        elif env_name == "moving_obstacles":
            # Failure Count
            if reward < 0:
                failure_count += 1
            if reward >= 0:
                avoid_count += 1
            if reward > 0:
                success_count += 1
                
            # Check what goal zone was reached
            if reward > 0:
                agent_x, agent_y = last_state[0], last_state[1]

                for n, region in enumerate(env.goal_region):
                    (x_min, y_min), (x_max, y_max) = region

                    if x_min <= agent_x <= x_max and y_min <= agent_y <= y_max:
                        goal_region = n
                goal_counts[goal_region] += 1

        all_rewards.append(current_reward)
        all_discounted_rewards.append(discounted_reward)

        if generate_gif and episode == 0 and len(frames) > 0:
            imageio.mimsave(gif_path, frames, fps=30)
     
     
    if env_name == "moving_obstacles":
        with open("goal_counts.txt", "a") as f:
            f.write(f"{gif_name} : {goal_counts}\n")           

    return (
        np.mean(all_rewards),
        np.mean(all_discounted_rewards),
        success_count / episode_count,
        failure_count / episode_count,
        avoid_count / episode_count
    )

