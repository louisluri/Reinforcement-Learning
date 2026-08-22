import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RescaleAction, ClipAction

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Layer(nn.Module):
    def __init__(self, input_size, output_size, activation="ReLU"):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        if activation == "ReLU":
            self.activation = nn.ReLU()
        if activation == "Tanh":
            self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.linear(x))
    
class DQN(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
        self.out = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        # Small initialization for the final output layer
        # This ensures starting q values are near 0
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)
    
class PG(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
        self.out = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        # Small initialization for the final output layer
        # This ensures initial actions are almost equally likely
        nn.init.uniform_(self.out.weight, -1e-2, 1e-2)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)
    
class PPO_Actor(nn.Module):
    def __init__(self, layer_sizes, continuous):
        super().__init__()
        self.continuous = continuous
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], activation="Tanh"))
        self.out = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        if self.continuous: self.log_std = nn.Parameter(torch.full((1, layer_sizes[-1]), -0.5))

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Actor Head: Very small gain so initial probabilities are nearly uniform
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.continuous:
            mean = self.out(x)
            log_std = self.log_std.expand_as(mean)
            std = torch.exp(log_std) # Convert log_std back to standard deviation
            return mean, std
        else:
            return self.out(x)
        
class PPO_Critic(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], activation="Tanh"))
        self.out = nn.Linear(layer_sizes[-2], 1)

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Critic Head: Standard gain of 1.0 for value estimation
        nn.init.orthogonal_(self.out.weight, gain=1.0)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)
        
class PPO(nn.Module):
    def __init__(self, layer_sizes, continuous):
        super().__init__()
        self.continuous = continuous
        self.actor = PPO_Actor(layer_sizes, continuous)
        self.critic = PPO_Critic(layer_sizes)

    def forward(self, x):
        if self.continuous:
            action_mean, action_std = self.actor(x)
            return action_mean, action_std, self.critic(x)
        else:
            return self.actor(x), self.critic(x)
        
class SAC_Actor(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
        self.mean_out = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.log_std_out = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                # sqrt(2) is the optimal gain for ReLU activations
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Small initialization for the final output layer
        # This ensures initial actions are centered and variance starts near 1.0
        nn.init.uniform_(self.mean_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_out.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_out.bias, -3e-3, 3e-3)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_out(x)
        log_std = self.log_std_out(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  
        std = torch.exp(log_std)
        return mean, std
    
class SAC_Critic(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            if i == 0: # Critic needs to take in the states plus the actions
                self.layers.append(Layer(layer_sizes[i] + layer_sizes[-1], layer_sizes[i + 1]))
            else:
                self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
        self.out = nn.Linear(layer_sizes[-2], 1) # and output one q value

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                # sqrt(2) is the optimal gain for ReLU activations
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Small initialization for the final output layer
        # This ensures starting q values are near 0
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.out.bias, -3e-3, 3e-3)

    def forward(self, x, action):
        sa = torch.cat((x, action), dim=-1)
        for layer in self.layers:
            sa = layer(sa)
        return self.out(sa)
    
class SAC_TC(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.c1 = SAC_Critic(layer_sizes)
        self.c2 = SAC_Critic(layer_sizes)

    def forward(self, x, action):
        return self.c1(x, action), self.c2(x, action)

class SAC(nn.Module):
    """just for saving the model after its been trained seperately, 
    this way the eval can see the action and the value together"""
    def __init__(self, actor=None, critic=None, layer_sizes=None):
        super().__init__()
        if layer_sizes is None:
            assert actor is not None and critic is not None, "SAC invalid args"
            self.actor = actor
            self.critic = critic
        else:
            assert actor is None and critic is None, "SAC invalid args"
            self.actor = SAC_Actor(layer_sizes)
            self.critic = SAC_TC(layer_sizes)

    def forward(self, x):
        action_mean = self.actor(x)[0]
        q1, q2 = self.critic(x, action_mean)
        return torch.tanh(action_mean), torch.min(q1, q2)

def save_weights(model, name):
    # Save only the weights (state_dict) to a .pt file
    torch.save(model.state_dict(), f"weights/{name}_weights.pt")
    print("Model weights saved successfully!")

def save_metrics(data, name, header):
    # Save to CSV, specifying a comma delimiter and a clean header
    np.savetxt(
        f"metrics/{name}_metrics.csv", 
        np.array(data), 
        delimiter=",", 
        header=header, 
        comments='' # Prevents NumPy from adding a '#' before the header
    )
    print(f"Metrics saved to {name}_metrics.csv!")

def save_norms(env, name):
    obs_rms = env.get_wrapper_attr("obs_rms")
    ret_rms = env.get_wrapper_attr("return_rms")
    os.makedirs("weights/norm_stats", exist_ok=True)
    np.savez(f"weights/norm_stats/{name}_norm_stats.npz", 
        obs_mean=obs_rms.mean, 
        obs_var=obs_rms.var, 
        reward_var=ret_rms.var)

    print("Env Normalization statistics saved successfully.")

def eval(env, model, device, cont=False, norm_stats={'obs_mean': 0, 'obs_var': 1, 'reward_var': 1}):
    state, _ = env.reset()
    episode_reward = 0
    total_reward = 0
    episode = 0
    obs_mean = norm_stats['obs_mean']
    obs_var = norm_stats['obs_var']

    model.eval()  # Set to evaluation mode
    with torch.no_grad(): # Dont waste resources on calculating gradients
        while True:
            state = (state - obs_mean) / np.sqrt(obs_var + 1e-8)
            logits = model(torch.from_numpy(state).float().unsqueeze(0).to(device))[0]
            if cont:
                action = logits.detach().cpu().numpy().flatten()
            else:
                action = torch.argmax(logits, dim=-1).item()

            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                episode += 1
                total_reward += episode_reward
                print(f"Episode {episode} complete! | Episode reward: {episode_reward:.4f}, Average reward: {total_reward / episode:.4f}")
                episode_reward = 0
                state, _ = env.reset()
    env.close()

def live_dashboard(env, model, device, action_labels, cont=False, norm_stats={'obs_mean': 0, 'obs_var': 1, 'reward_var': 1}):
    obs_mean = norm_stats['obs_mean']
    obs_var = norm_stats['obs_var']
    reward_var = norm_stats['reward_var']

    layout = """
    AD
    AB
    AC
    AC
    AC
    """
    fig, axs = plt.subplot_mosaic(layout, figsize=(12, 6), facecolor="#456F72")
    
    state, _ = env.reset()
    img = axs['A'].imshow(env.render())
    axs['A'].axis('off')
    
    bars = axs['B'].bar(action_labels, [0] * len(action_labels), color="#ccc671")
    axs['B'].set_ylim(-1.1, 1.1)
    axs['B'].set_title(f"Model Action Outputs | Continuous: {cont}")
    axs['B'].axhline(0, color='black', lw=1) # Zero line
    axs['B'].set_facecolor("#616161")

    axs['C'].axis('off')
    value_text = axs['C'].text(0.2, 0.6, f"Value = {0}\nTotal Reward = {0}\nAverage Reward = {0}", 
            ha='left', va='center', 
            fontsize=15, color='black', fontweight="bold",
            transform=axs['C'].transAxes)
    
    axs["D"].axis('off')

    episode_reward = 0
    total_reward = 0
    episode = 0
    def update(frame):
        nonlocal state, episode_reward, total_reward, episode
        
        norm_state = (state - obs_mean) / np.sqrt(obs_var + 1e-8)
        state_t = torch.from_numpy(norm_state).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(state_t)
            logits, value = output[0], output[-1]
            action = logits.squeeze().cpu().numpy()

        if cont:
            action_to_take = action
        else:
            action_to_take = np.argmax(action)
            max_abs = np.max(np.abs(action))
            action = action / (max_abs + 1e-8)

        state, reward, terminated, truncated, _ = env.step(action_to_take)
        
        episode_reward += reward
        value_text.set_text(f"Value = {value.max().item() * np.sqrt(reward_var + 1e-8):.2f}\nTotal Reward = {episode_reward:.2f}\nAverage Reward = {total_reward / (episode + 1e-12):.2f}")
        img.set_data(env.render())
        
        for bar, val in zip(bars, action):
            bar.set_height(val)
            
        if terminated or truncated:
            state, _ = env.reset()
            episode += 1
            total_reward += episode_reward
            print(f"Episode {episode} complete! | Episode reward: {episode_reward:.4f}, Average reward: {total_reward / episode:.4f}")
            episode_reward = 0
            
        return [img, value_text] + list(bars)

    # interval=20 matches roughly 50fps
    ani = FuncAnimation(fig, update, interval=5, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

def record_video(model, game, device, save_name, step):
    """only set up for sac training right now"""

    print(f"Recording video evaluation at step {step}...")

    video_env = gym.make(game, render_mode="rgb_array")

    video_env = RescaleAction(video_env, min_action=-1.0, max_action=1.0)
    video_env = ClipAction(video_env)

    video_env = RecordVideo(
        video_env, 
        video_folder=f"./videos/{save_name}", 
        name_prefix=f"eval_step_{step}",
        episode_trigger=lambda x: True  # Only record episode 0!
    )
    
    eval_state, _ = video_env.reset()
    eval_reward = 0
    eval_over = False

    model.eval()

    while not eval_over:
        with torch.no_grad():
            action_mean, _ = model(torch.from_numpy(eval_state).float().unsqueeze(0).to(device))
            eval_action = torch.tanh(action_mean).cpu().numpy()[0] 
        
        eval_state, reward, terminated, truncated, _ = video_env.step(eval_action)
        eval_reward += reward
        if terminated or truncated:
            eval_over = True
    
    # close the env to flush the mp4 file to your disk
    video_env.close()
    model.train()

    print(f"Video saved! Evaluation reward: {eval_reward:.2f}")  
    return eval_reward

def get_game(game_code):
    if game_code == "ll":
        game = "LunarLander-v3"
        action_labels = ["None", "Left", "Main", "Right"]
    elif game_code == "llcont":
        game = "LunarLanderContinuous-v3"
        action_labels = ["Main", "Side"]
    elif game_code == "walk":
        game = "Walker2d-v5"
        action_labels = ["r_thigh", "r_leg", "r_foot", "l_thigh", "l_leg", "l_foot"]
    elif game_code == "bw":
        game = "BipedalWalker-v3"
        action_labels = ["0", "1", "2", "3"]
    elif game_code == "chee":
        game = "HalfCheetah-v5"
        action_labels = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
    elif game_code == "human":
        game = "Humanoid-v5"
        action_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
    else:
        raise ValueError("Invalid game code")
    
    print(f"{game} selected")
    return game, action_labels