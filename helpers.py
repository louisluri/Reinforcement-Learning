import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RescaleAction, ClipAction

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

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