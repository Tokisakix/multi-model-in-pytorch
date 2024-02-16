import torch
import numpy as np
import torch.nn.functional as F

from load_config import load_config

class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = torch.nn.Parameter(torch.empty(out_features))
        self.bias_sigma = torch.nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()
        return

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
        return
    
    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.epsilon_weight = epsilon_out.ger(epsilon_in)
        self.epsilon_bias = self.scale_noise(self.out_features)
        return

    def forward(self, input, training):
        if training:
            weight_epsilon = self.weight_sigma * self.epsilon_weight.to(input.device)
            weight = self.weight_mu + weight_epsilon
            bias_epsilon = self.bias_sigma * self.epsilon_bias.to(input.device)
            bias = self.bias_mu + bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

    def scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

class Model(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, num_atoms=10):
        super().__init__()
        self.action_dim = action_dim
        self.num_atoms  = num_atoms
        self.fc1 = NoisyLinear(state_dim, hidden_dim)
        self.V   = NoisyLinear(hidden_dim, action_dim)
        self.A   = NoisyLinear(hidden_dim, action_dim * num_atoms)
        return

    def forward(self, x, training=True):
        x = F.relu(self.fc1(x, training))
        v = self.V(x, training).reshape(-1, self.action_dim, 1)
        a = self.A(x, training).reshape(-1, self.action_dim, self.num_atoms)
        x = v + a - a.mean(dim=-1, keepdim=True)
        x = F.softmax(x, dim=-1)
        return x



# ---TEST---

if __name__ == "__main__":
    CONFIG  = load_config()
    CUDA    = CONFIG["cuda"]

    inputs  = torch.randn(32, 4)
    model   = Model(state_dim=4, hidden_dim=128, action_dim=2)

    inputs  = inputs.cuda() if CUDA else inputs
    model   = model.cuda() if CUDA else model
    outputs = model(inputs)

    print(inputs.shape)
    print(model.eval())
    print(outputs.shape)