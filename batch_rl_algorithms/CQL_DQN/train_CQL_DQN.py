import numpy as np
import random 
import torch
from .buffer import ReplayBuffer
from .agent import CQLAgent
from collections import deque



def add_dataset_to_buffer(dataset, buffer_size, device):
    buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=32, device=device)
    # state, action_choice, next_state, reward, terminated, truncated
    for traj in dataset:
        for trans in traj:
            state = trans[0]
            action = trans[1]
            next_state = trans[2]
            reward = trans[3]
            is_done = trans[4] or trans[5]
            buffer.add(state, action, reward, next_state, is_done)
    
    return buffer

def train_cql_dqn(env, dataset, buffer_size=32, nb_iterations=10000, verbose=False):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the agent
    agent = CQLAgent(state_size=env.get_state_shape(),
                        action_size=env.get_nb_actions(),
                        device=device)

    # Load dataset into replay buffer
    buffer = add_dataset_to_buffer(dataset, buffer_size, device)

    losses = []
    cql_losses = []
    bellman_losses = []

    # Offline training loop
    for step in range(nb_iterations):

        batch = buffer.sample()

        loss, cql_loss, bellmann_error = agent.learn(batch)

        losses.append(loss)
        cql_losses.append(cql_loss)
        bellman_losses.append(bellmann_error)

        if step % 100 == 0 and verbose:
            print(
                f"Step {step} | "
                f"Total Loss: {loss:.4f} | "
                f"CQL Loss: {cql_loss:.4f} | "
                f"Bellman Error: {bellmann_error:.4f}"
            )

    return agent

def train_cql_dqn_hybrid(env, dataset, buffer_size, nb_itterations, eps_frames=1e-4, min_eps=0.01):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create the agent
    agent = CQLAgent(state_size=env.get_state_shape(),
                        action_size=env.get_nb_actions(),
                        device=device)

    # Format our data to match their buffer
    buffer = add_dataset_to_buffer(dataset, buffer_size, device)
    
    # Training Loop
    eps = 1.
    d_eps = 1 - min_eps
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0
    for i in range(1, nb_itterations):
        env.reset()
        state = env.get_state()
        episode_steps = 0
        rewards = 0
        while True:
            action = agent.get_action(state, epsilon=eps)
            steps += 1
            _  ,next_state, reward = env.step(action[0])
            done = env.is_done()
            # buffer.add(state, action, reward, next_state, done)
            loss, cql_loss, bellmann_error = agent.learn(buffer.sample())
            state = next_state
            rewards += reward
            episode_steps += 1
            eps = max(1 - ((steps*d_eps)/eps_frames), min_eps)
            if done:
                break

        average10.append(rewards)
        total_steps += episode_steps
        print("Episode: {} | Reward: {} | Q Loss: {} | Steps: {}".format(i, rewards, loss, steps,))
        
    return agent
            
