import torch
import torch.nn as nn
import numpy as np

class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.linear(x))
    
class PPO_Actor(nn.Module):
    def __init__(self, state_sizes, continuous):
        super().__init__()
        self.continuous = continuous
        self.layers = nn.ModuleList()
        for i in range(len(state_sizes) - 2):
            self.layers.append(Layer(state_sizes[i], state_sizes[i + 1]))
        self.out = nn.Linear(state_sizes[-2], state_sizes[-1])
        if self.continuous: self.log_std = nn.Parameter(torch.full((1, state_sizes[-1]), -0.5))

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
    def __init__(self, state_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(state_sizes) - 2):
            self.layers.append(Layer(state_sizes[i], state_sizes[i + 1]))
        self.out = nn.Linear(state_sizes[-2], 1)

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
    def __init__(self, state_sizes, continuous):
        super().__init__()
        self.continuous = continuous
        self.actor = PPO_Actor(state_sizes, continuous)
        self.critic = PPO_Critic(state_sizes)

    def forward(self, x):
        if self.continuous:
            action_mean, action_std = self.actor(x)
            return action_mean, action_std, self.critic(x)
        else:
            return self.actor(x), self.critic(x)