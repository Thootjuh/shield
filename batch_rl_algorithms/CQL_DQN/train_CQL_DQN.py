import numpy as np
import random 
import torch
from .buffer import ReplayBuffer
from .agent import CQLAgent
from collections import deque
from torch.utils.data import DataLoader, TensorDataset


def add_dataset_to_buffer(dataset, buffer_size, device):
    buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=32, device=device)
    # state, action_choice, next_state, reward, terminated, truncated
    for trans in dataset:
        # for trans in traj:
        state = trans[0]
        action = trans[1]
        next_state = trans[2]
        reward = trans[3]
        is_done = trans[4] or trans[5]
        buffer.add(state, action, reward, next_state, is_done)
    
    return buffer
def create_dataloader_from_dataset(dataset, batch_size, device):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for trans in dataset:
        state = trans[0]
        action = trans[1]
        next_state = trans[2]
        reward = trans[3]
        done = trans[4] or trans[5]   # terminated OR truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
    # Convert to tensors
    states      = torch.tensor(np.array(states), dtype=torch.float32)
    actions     = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1)
    rewards     = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones       = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)

    dataset_torch = TensorDataset(states, actions, rewards, next_states, dones)
    dataloader = DataLoader(dataset_torch, batch_size=batch_size, shuffle=True)

    return dataloader

def train_cql_dqn(
    env,
    env_name,
    dataset_raw,
    nb_epochs=100,
    batch_size=64
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = [val for sublist in dataset_raw for val in sublist]
    if env_name=='lunar_lander':
        max_updates=500000
    else:
        max_updates=100000
    updates_per_epoch = len(dataset) // batch_size
    nb_epochs = min(max(1, max_updates // updates_per_epoch),10)
    # nb_epochs = 10
    print(len(dataset))
    print(nb_epochs)
    # Agent
    agent = CQLAgent(
        state_size=env.get_state_shape(),
        action_size=env.get_nb_actions(),
        device=device
    )

    # DataLoader
    dataloader = create_dataloader_from_dataset(dataset, batch_size, device)

    # Training loop
    total_updates = 0
    avg_loss = deque(maxlen=100)

    for epoch in range(1, nb_epochs + 1):

        for batch in dataloader:
            states, actions, rewards, next_states, dones = [
                x.to(device) for x in batch
            ]

            # Same interface as ReplayBuffer.sample()
            loss, cql_loss, bellmann_error = agent.learn(
                (states, actions, rewards, next_states, dones)
            )

            total_updates += 1
            avg_loss.append(loss)

        print(
            f"Epoch {epoch} | "
            f"Total Loss: {loss:.4f} | "
            f"CQL Loss: {cql_loss:.4f} | "
            f"Bellman Error: {bellmann_error:.4f}"
        )

    return agent
                              
def train_cql_dqn_buffer(env, dataset_raw, buffer_size=32, Passes_on_dataset=20, verbose=False):
    dataset = [val for sublist in dataset_raw for val in sublist]
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
    total_steps = 0
    # Offline training loop
    for ds_pass in range(Passes_on_dataset):
        steps=0
        while steps < len(dataset):
            batch = buffer.sample()
            steps += buffer_size
            total_steps += buffer_size
            loss, cql_loss, bellmann_error = agent.learn(batch)

            losses.append(loss)
            cql_losses.append(cql_loss)
            bellman_losses.append(bellmann_error)

        # if step % 100 == 0 and verbose:
        print(
            f"Passes {ds_pass} | "
            f"Step {total_steps} | "
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
            
