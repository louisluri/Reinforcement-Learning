import torch
import torch.nn as nn
import numpy as np

class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))
        
class SAC_Actor(nn.Module):
    def __init__(self, state_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(state_sizes) - 2):
            self.layers.append(Layer(state_sizes[i], state_sizes[i + 1]))
        self.mean_out = nn.Linear(state_sizes[-2], state_sizes[-1])
        self.log_std_out = nn.Linear(state_sizes[-2], state_sizes[-1])

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
    def __init__(self, state_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(state_sizes) - 2):
            if i == 0: # Critic needs to take in the states plus the actions
                self.layers.append(Layer(state_sizes[i] + state_sizes[-1], state_sizes[i + 1]))
            else:
                self.layers.append(Layer(state_sizes[i], state_sizes[i + 1]))
        self.out = nn.Linear(state_sizes[-2], 1) # and output one q value

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
    def __init__(self, state_sizes):
        super().__init__()
        self.c1 = SAC_Critic(state_sizes)
        self.c2 = SAC_Critic(state_sizes)

    def forward(self, x, action):
        return self.c1(x, action), self.c2(x, action)

class SAC(nn.Module):
    """just for saving the model after its been trained seperately, 
    this way the eval can see the action and the value together"""
    def __init__(self, actor=None, critic=None, state_sizes=None):
        super().__init__()
        if state_sizes is None:
            assert actor is not None and critic is not None, "SAC invalid args"
            self.actor = actor
            self.critic = critic
        else:
            assert actor is None and critic is None, "SAC invalid args"
            self.actor = SAC_Actor(state_sizes)
            self.critic = SAC_TC(state_sizes)

    def forward(self, x):
        action_mean = self.actor(x)[0]
        q1, q2 = self.critic(x, action_mean)
        return torch.tanh(action_mean), torch.min(q1, q2)