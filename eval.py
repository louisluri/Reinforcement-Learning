import gymnasium as gym
from gymnasium.wrappers import RescaleAction, ClipAction
import numpy as np
import torch 
import os
from model import DQN, PG, PPO, SAC, eval, live_dashboard, get_game

device = torch.device("cuda:1")

game_code = "llcont"
model_code = "ppo"
model_number = "1"
hidden = 64
model_name = f"{game_code}_{model_code}{model_number}"

dashboard = 0

game, action_labels = get_game(game_code)

if dashboard:
  env = gym.make(game, render_mode="rgb_array")
else:
  env = gym.make(game, render_mode=None)

if isinstance(env.action_space, gym.spaces.Discrete):
    print("Action Space is Discrete")
    continuous = False
elif isinstance(env.action_space, gym.spaces.Box):
    print("Action Space is Continuous")
    continuous = True
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    env = ClipAction(env)
else:
    raise ValueError(f"Action Space is a different type: {type(env.action_space)}")

layers = [env.observation_space.shape[0], hidden, hidden, env.action_space.shape[0] if continuous else env.action_space.n]  # input size, hidden layers, output size

if model_code == "sac":
  m = SAC(layer_sizes=layers).to(device)
elif model_code == "ppo":
  m = PPO(layers, continuous=continuous).to(device)
elif model_code == "pg":
  m = PG(layers).to(device)
elif model_code == "dqn":
  m = DQN(layers).to(device)
else:
  raise ValueError("Invalid model code")

state_dict = torch.load(f"weights/{model_name}_weights.pt")
m.load_state_dict(state_dict)

if os.path.exists(f"weights/norm_stats/{model_name}_norm_stats.npz"):
  norm_stats = np.load(f"weights/norm_stats/{model_name}_norm_stats.npz")
  print("Loaded norms sucessfully")
else:
  norm_stats = {'obs_mean': 0, 'obs_var': 1, 'reward_var': 1}
  print("No norms found")

if dashboard:
  live_dashboard(env, m, device, action_labels=action_labels, cont=continuous, norm_stats=norm_stats)
else:
  eval(env, m, device, cont=continuous, norm_stats=norm_stats)