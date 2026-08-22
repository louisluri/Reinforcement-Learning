# polocy based method, learns a policy directly instead of learning a value function like DQN
# REINFORCE algorithm, Monte Carlo policy gradient method

# objective function is the expected return of the policy, we want to maximize this

# Objective Function: J(θ)=Eτ∼πθ​​[R(τ)] ,E is the expected value, J(θ)=τ∑ ​P(τ∣θ) R(τ)
# A weighted sum of all the rewards if a trajectory is followed, based on how probable that tragectory is given the current policy

# Policy Gradient Theorem: ∇θ​J(θ)=Eτ∼πθ​​[t=0∑T ​∇θ​logπθ​(at​∣st​)⋅Gt​]
# grad F(theta) = F(theta) * grad log F(theta)

# the final weighted sum part gets canceled by the fact that the episodes are being sampled 
# according to the current policy

import gymnasium as gym
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
from model import PG, save_weights, save_metrics, get_game

device = torch.device("cuda:1")
        
game_code = "ll"
model_code = "pg"
model_number = "1"
save_name = f"{game_code}_{model_code}{model_number}"

# Hyperparameters
batch_size = 2000 # how many target steps per update
train_steps = 2000000
gamma = 0.99  # Discount factor
lr = 0.0005  # Learning rate

game = get_game(game_code)[0]    
env = gym.make(game, render_mode=None)

assert isinstance(env.action_space, gym.spaces.Discrete), "Action Space is not discrete"

model_layers = [env.observation_space.shape[0], 64, 64, env.action_space.n]  # input size, hidden layers, output size

m = PG(model_layers).to(device)

o = optim.Adam(m.parameters(), lr=lr)

state, info = env.reset()
total_reward = 0
episode = 0

saved_log_probs = [] # To store log probabilities of actions taken, used for calculating the policy gradient
rewards = [] # To store rewards for each episode, used for calculating returns
saved_returns = [] # To store the discounted returns for each episode, used for calculating the policy gradient

step_counter = 0
history = [] # To store episode, step, reward for plotting
for step in range(train_steps):
    logits = m(torch.from_numpy(state).float().unsqueeze(0).to(device))
    probs = F.softmax(logits, dim=-1)

    # Create a distribution object 
    # This "wraps" the probabilities but keeps the gradient link to the model weights
    dist = Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action) # gradient string stays attatched
    
    saved_log_probs.append(log_prob)

    state, reward, terminated, truncated, info = env.step(action.item())
    rewards.append(reward)

    total_reward += reward

    step_counter += 1
    if terminated or truncated:
        state, info = env.reset()
        episode += 1
        history.append([episode, step, total_reward])
        print(f"Episode {episode} complete! | Step: {step}, Total reward: {total_reward:.4f}")
        total_reward = 0

        # Calculate discounted returns
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)  # Insert at the beginning to maintain order
        saved_returns.extend(returns)
        rewards = [] # Clear rewards for the next episode

        if step_counter >= batch_size:
            returns = torch.tensor(saved_returns).to(device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            log_probs = torch.cat(saved_log_probs)

            loss = -(log_probs * returns).sum() # Eτ∼πθ​​[t=0∑T ​∇θ​logπθ​(at​∣st​)⋅Gt​] (negative for gradient descent)
            o.zero_grad()
            loss.backward()
            o.step()

            saved_log_probs = []
            saved_returns = []
            step_counter = 0


print("Training complete!")

save_weights(m, save_name)
save_metrics(history, save_name, "episode,step,reward")

env.close()