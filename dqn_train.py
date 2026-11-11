import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import copy
import random
import numpy as np
from dqn_model import DQN
from helpers import save_weights, save_metrics, get_game

device = torch.device("cuda:1")

def e_greedy_action_selection(action_values, eps):
    if random.random() < eps:
        return random.randint(0, action_values.size(-1) - 1)  # Explore
    else:
        return torch.argmax(action_values, dim=-1).item()  # Exploit

def soft_update(model, target_model, tau):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

game_code = "ll"
model_code = "dqn"
model_number = "1"
save_name = f"{game_code}_{model_code}{model_number}"

memory = deque(maxlen=50000)

# Hyperparameters
batch_size = 128
train_steps = 200000
gamma = 0.99  # Discount factor
lr = 0.001  # Learning rate
eps_max = 0.9  # Exploration rate
eps_min = 0.05
tau = 0.005 # Small value, e.g., 0.5% update

game = get_game(game_code)[0]    
env = gym.make(game, render_mode=None)

assert isinstance(env.action_space, gym.spaces.Discrete), "Action Space is not discrete"

model_layers = [env.observation_space.shape[0], 64, 64, env.action_space.n]  # input size, hidden states, output size

m = DQN(model_layers).to(device)
target_m = copy.deepcopy(m).to(device)

# Ensure the target model is in eval mode and doesn't track gradients
target_m.eval()
for param in target_m.parameters():
    param.requires_grad = False

o = optim.Adam(m.parameters(), lr=lr)

state, info = env.reset()
total_reward = 0
episode = 0
history = [] # To store episode, step, epsilon, reward for plotting
for step in range(train_steps):
    current_eps = max(eps_min, eps_max - (eps_max - eps_min) * (step / (train_steps * 0.5)))  # Linear decay of epsilon

    m.eval()  # Set to evaluation mode
    with torch.no_grad(): # Dont waste resources on calculating gradients
        q_values = m(torch.from_numpy(state).float().unsqueeze(0).to(device))
    m.train()
    
    action = e_greedy_action_selection(q_values, current_eps)

    next_state, reward, terminated, truncated, info = env.step(action)

    memory.append((state, action, reward, next_state, terminated))
    total_reward += reward
    state = next_state

    if terminated or truncated:
        state, info = env.reset()
        episode += 1
        diff = torch.norm(torch.stack([torch.norm(p1 - p2) for p1, p2 in zip(m.parameters(), target_m.parameters())]))
        history.append([episode, step, current_eps, total_reward])
        print(f"Episode {episode} complete! | Step: {step}, Epsilon (ε): {current_eps:.4f}, Diff: {diff:.4f}, Total reward: {total_reward:.4f}")
        total_reward = 0

    # train on data thats in memory
    if step >= 1000:
        batch = random.sample(memory, batch_size)

        states, actions, rewards, next_states, terminateds = zip(*batch)

        state_batch = torch.from_numpy(np.array(states)).float()
        action_batch = torch.tensor(actions).unsqueeze(1) # Needs to be [[a1], [a2]...]
        reward_batch = torch.tensor(rewards).float()
        next_state_batch = torch.from_numpy(np.array(next_states)).float()
        terminal_batch = torch.tensor(terminateds).float()

        q_values = m(state_batch.to(device)).gather(1, action_batch.to(device)).squeeze(1)

        target_m.eval()
        with torch.no_grad():
            next_q_values = target_m(next_state_batch.to(device)).max(1)[0]

        target_qs = reward_batch.clone().to(device)
        non_final_mask = (terminal_batch == 0)
        target_qs[non_final_mask] += gamma * next_q_values[non_final_mask]

        l = F.smooth_l1_loss(q_values, target_qs) # Huber loss, less sensitive to outliers than MSE
        o.zero_grad()
        l.backward()
        o.step()

        soft_update(m, target_m, tau)

print("Training complete!")

save_weights(m, save_name)
save_metrics(history, save_name, "episode,step,epsilon,reward")

env.close()