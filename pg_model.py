import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))
    
class PG(nn.Module):
    def __init__(self, state_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(state_sizes) - 2):
            self.layers.append(Layer(state_sizes[i], state_sizes[i + 1]))
        self.out = nn.Linear(state_sizes[-2], state_sizes[-1])

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