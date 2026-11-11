# SAC Soft Actor Critic model
# Off-Policy method with a twin critic archetecture

# Objective: ∑ γt E[rt​+αH(π(⋅∣st​))]
# Target Q = r+γ(1−d)(min(​Qθ​​(s′,a′)) − αlogπϕ​(a′∣s′))
# L_actor ​= mean(αlogπϕ​(a~∣s) − min(​Qθi​​(s,a~)))
# L(α) = −α × (logπt​(at​∣st​) + Htarget​)

import gymnasium as gym
from gymnasium.wrappers import RescaleAction, ClipAction
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from collections import deque
import copy
import random
import numpy as np
from sac_model import SAC, SAC_Actor, SAC_TC
from helpers import save_weights, save_metrics, get_game, record_video

device = torch.device("cuda:1")

def soft_update(model, target_model, tau):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

game_code = "human"
model_code = "sac"
model_number = "1"
hidden = 512
save_name = f"{game_code}_{model_code}{model_number}"

memory = deque(maxlen=1000000)

# Hyperparameters
batch_size = 256
train_steps = 10000000
gamma = 0.99    # Discount factor
lr = 0.0003     # Learning rate
tau = 0.005     # Small value, e.g., 0.5% update
warmup = 10000
eval_episodes = 3  # how any evaluation episodes
eval_interval = 100000 # how often to eval

game = get_game(game_code)[0]
env = gym.make(game, render_mode=None)

assert isinstance(env.action_space, gym.spaces.Box), "Action Space is not continuous"

model_layers = [env.observation_space.shape[0], hidden, hidden, env.action_space.shape[0]]  # input size, hidden states, output size

m = SAC_Actor(model_layers).to(device)
c = SAC_TC(model_layers).to(device)
target_c = copy.deepcopy(c).to(device)

log_a = torch.tensor([0.0], requires_grad=True, device=device)
target_entropy = -float(env.action_space.shape[0])

# Ensure the target critic is in eval mode and doesn't track gradients
target_c.eval()
for param in target_c.parameters():
    param.requires_grad = False

mo = optim.Adam(m.parameters(), lr=lr)
co = optim.Adam(c.parameters(), lr=lr)
ao = optim.Adam([log_a], lr=lr)

env = RescaleAction(env, min_action=-1.0, max_action=1.0)
env = ClipAction(env)

state, info = env.reset()
total_reward = 0
episode = 0
train_history = [] # To store episode, step, reward for plotting
alpha = log_a.exp().detach()
current_mean_std = 1
record_video(m, game, device, save_name, 0)
for step in range(train_steps):

    if step < warmup:
        # Perfectly uniform distribution
        action = env.action_space.sample() 
    else:
        m.eval()
        with torch.no_grad():
            action_mean, action_std = m(torch.from_numpy(state).float().unsqueeze(0).to(device))
            dist = Normal(action_mean, action_std)
            action = torch.tanh(dist.sample()).cpu().numpy()[0]
        m.train()

    next_state, reward, terminated, truncated, info = env.step(action)

    memory.append((state, action, reward, next_state, terminated))
    total_reward += reward
    state = next_state

    if terminated or truncated:
        state, info = env.reset()
        episode += 1
        diff = torch.norm(torch.stack([torch.norm(p1 - p2) for p1, p2 in zip(c.parameters(), target_c.parameters())]))
        train_history.append([episode, step, total_reward])
        print(f"Episode {episode} complete! | Step: {step} | Diff: {diff:.4f} | Alpha: {alpha.item():.4f} | Total reward: {total_reward:.4f} | std: {current_mean_std:.4f}")
        total_reward = 0

    # train on data thats in memory
    if step >= warmup:
        alpha = log_a.exp().detach()
        batch = random.sample(memory, batch_size)

        states, actions, rewards, next_states, terminateds = zip(*batch)

        state_batch = torch.from_numpy(np.array(states)).float().to(device)
        action_batch = torch.from_numpy(np.array(actions)).float().to(device)
        reward_batch = torch.tensor(rewards).float().to(device)
        next_state_batch = torch.from_numpy(np.array(next_states)).float().to(device)
        terminal_batch = torch.tensor(terminateds).float().to(device)

        m.eval()
        with torch.no_grad():
            next_action_mean, next_action_std = m(next_state_batch)
        m.train()

        current_mean_std = next_action_std.mean()

        dist = Normal(next_action_mean, next_action_std)
        next_action = dist.sample()
        next_log_probs = dist.log_prob(next_action).sum(dim=-1, keepdim=True)
        next_log_probs -= (2 * (np.log(2) - next_action - F.softplus(-2 * next_action))).sum(dim=-1, keepdim=True)
        next_action = torch.tanh(next_action)

        target_c.eval()
        with torch.no_grad():
            next_q1, next_q2 = target_c(next_state_batch, next_action)

        next_q_values = torch.min(next_q1, next_q2)
        target_qs = reward_batch.view(-1, 1) + gamma * (1 - terminal_batch.view(-1, 1)) * (next_q_values - alpha * next_log_probs)

        q1, q2 = c(state_batch, action_batch)

        cl = F.mse_loss(q1, target_qs) + F.mse_loss(q2, target_qs)
        co.zero_grad()
        cl.backward()
        co.step()

        action_mean, action_std = m(state_batch)

        dist = Normal(action_mean, action_std)
        action = dist.rsample()
        log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        log_probs -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1, keepdim=True)
        action = torch.tanh(action)

        q1, q2 = c(state_batch, action)

        ml = (alpha * log_probs - torch.min(q1, q2)).mean()
        mo.zero_grad()
        ml.backward()
        mo.step()

        soft_update(c, target_c, tau)

        entropy_diff = log_probs.detach() + target_entropy
        al = -(log_a.exp() * entropy_diff).mean()
        ao.zero_grad()
        al.backward()
        ao.step()

        #print(f"Model loss: {ml:.3f}, Critic loss: {cl:.3f}, Alpha loss: {al:.3f}")

    if (step+1) % eval_interval == 0:
        record_video(m, game, device, save_name, step+1)

print("Training complete!")

model = SAC(actor=m, critic=c).to(device)
save_weights(model, save_name)
save_metrics(train_history, save_name, "episode,step,reward")

env.close()