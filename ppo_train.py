# PPO "Proximal Policy Optimization"
# Clipped Surrogate Objective Function: LCLIP(θ)=Et​[min(rt​(θ)A^t​,clip(rt​(θ),1−ϵ,1+ϵ)A^t​)]

import gymnasium as gym
import torch
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ppo_model import PPO
from helpers import save_weights, save_metrics, save_norms, get_game

device = torch.device("cuda:1")
    
class RolloutBuffer():
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def store(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, next_value, gamma, gae_lambda):
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32).squeeze()
        dones = torch.tensor(self.dones, dtype=torch.float32)
        
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        buffer_size = len(rewards)
        for t in reversed(range(buffer_size)):
            if t == buffer_size - 1:
                # We must look outside the buffer at the bootstrap value
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]
                
            # TD Error
            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            
            # Accumulate the Advantage backwards
            advantages[t] = last_gae_lam = delta + (gamma * gae_lambda * next_non_terminal * last_gae_lam)
            
        self.advantages = advantages
        self.target_returns = advantages + values

        # Convert all to tensors before training
        self.states = torch.tensor(np.array(self.states), dtype=torch.float32).to(device)    
        self.actions = torch.tensor(np.array(self.actions)).to(device)

        self.log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        self.values = torch.tensor(self.values, dtype=torch.float32).to(device)

        self.advantages = self.advantages.clone().detach().to(device)
        self.target_returns = self.target_returns.clone().detach().to(device)
        
    def get_mini_batches(self, batch_size, minibatch_size):
        indices = torch.randperm(batch_size)

        for start_idx in range(0, batch_size, minibatch_size):
            mb_indices = indices[start_idx : start_idx + minibatch_size]
            
            mb_states = self.states[mb_indices]
            mb_actions = self.actions[mb_indices]
            mb_old_log_probs = self.log_probs[mb_indices]
            mb_advantages = self.advantages[mb_indices]
            mb_targets = self.target_returns[mb_indices]
            mb_old_values = self.values[mb_indices]
            
            yield mb_states, mb_actions, mb_old_log_probs, mb_advantages, mb_targets, mb_old_values

class TrackRealReward(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Store the real, unscaled reward in the info dict
        info["real_reward"] = reward
        return obs, reward, terminated, truncated, info

game_code = "llcont"
model_code = "ppo"
model_number = "2"
save_name = f"{game_code}_{model_code}{model_number}"

# Hyperparameters
batch_size = 2048
minibatch_size = 64 # how many target steps per update
num_epochs = 10
train_steps = 500000
gamma = 0.99  # Discount factor
gae_lambda = 0.95
lr = 0.0003  # Learning rate
    
game = get_game(game_code)[0]
env = gym.make(game, render_mode=None)

if isinstance(env.action_space, gym.spaces.Discrete):
    print("Action Space is Discrete")
    continuous = False
elif isinstance(env.action_space, gym.spaces.Box):
    print("Action Space is Continuous")
    continuous = True
    env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
    env = gym.wrappers.ClipAction(env)
else:
    raise ValueError(f"Action Space is a different type: {type(env.action_space)}")

model_layers = [env.observation_space.shape[0], 64, 64, env.action_space.shape[0] if continuous else env.action_space.n]  # input size, hidden layers, output size

m = PPO(model_layers, continuous=continuous).to(device)

o = optim.Adam(m.parameters(), lr=lr, eps=1e-5)

env = gym.wrappers.NormalizeObservation(env)
env = TrackRealReward(env)
env = gym.wrappers.NormalizeReward(env)
        
state, info = env.reset()
total_reward = 0
episode = 0
buffer = RolloutBuffer()
history = [] # To store episode, step, reward for plotting
for step in range(train_steps):
    with torch.no_grad():
        output = m(torch.from_numpy(state).float().unsqueeze(0).to(device))

    if continuous:
        dist = Normal(output[0], output[1])
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        new_state, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])

        # Extract to pure NumPy/Python types
        raw_action = action.cpu().numpy()[0]
        raw_log_prob = log_prob.item()
    else:
        probs = F.softmax(output[0], dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        new_state, reward, terminated, truncated, info = env.step(action.item())

        raw_action = action.item() 
        raw_log_prob = log_prob.item()

    if truncated:
        real_final_state_tensor = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad(): # find what the critic thinks the rest of the truncated run 
            # will be worth and preemptively add it
            bootstrap_value = m(real_final_state_tensor)[-1]
            
        reward = reward + (gamma * bootstrap_value.item())
        terminated = True

    buffer.store(state, raw_action, reward, raw_log_prob, output[-1].item(), terminated)

    total_reward += info["real_reward"]
    state = new_state

    if terminated or truncated:
        state, info = env.reset()
        episode += 1
        history.append([episode, step, total_reward])
        print(f"Episode {episode} complete! | Step: {step}, Total reward: {total_reward:.4f}")
        total_reward = 0


    if (step+1) % batch_size == 0:
        with torch.no_grad():
            next_value = m(torch.from_numpy(state).float().unsqueeze(0).to(device))[-1]

        buffer.compute_gae(next_value.item(), gamma, gae_lambda)
        buffer.advantages = (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

        for epoch in range(num_epochs):
            for mb_data in buffer.get_mini_batches(batch_size, minibatch_size):
                mb_states, mb_actions, mb_old_log_probs, mb_advs, mb_targets, mb_old_values = mb_data

                new_output = m(mb_states)
                new_values = new_output[-1].squeeze()

                # Create a distribution object 
                # This "wraps" but keeps the gradient link to the model weights
                if continuous:
                    dist = Normal(new_output[0], new_output[1])
                    new_log_probs = dist.log_prob(mb_actions).sum(dim=-1) # gradient string stays attatched
                    entropy_bonus = dist.entropy().sum(dim=-1).mean()
                else:
                    probs = F.softmax(new_output[0], dim=-1)
                    dist = Categorical(probs)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy_bonus = dist.entropy().mean()

                r = (new_log_probs - mb_old_log_probs).exp()
                unclipped_actor = r * mb_advs
                clipped_actor = torch.clamp(r, 0.8, 1.2) * mb_advs

                actor_loss = -torch.min(unclipped_actor, clipped_actor).mean()

                critic_loss = F.mse_loss(new_values, mb_targets)
                
                total_loss = actor_loss + (0.5 * critic_loss) - (0.01 * entropy_bonus)

                o.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=0.5)

                o.step()

        buffer.clear()


print("Training complete!")

save_weights(m, save_name)
save_metrics(history, save_name, "episode,step,reward")
save_norms(env, save_name)

env.close()