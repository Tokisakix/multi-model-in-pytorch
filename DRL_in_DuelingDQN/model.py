import torch
import torch.nn.functional as F

from load_config import load_config

class Model(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.V = torch.nn.Linear(hidden_dim, 1)
        self.A = torch.nn.Linear(hidden_dim, action_dim)
        return

    def forward(self, x):
        x = F.relu(self.fc1(x))
        v = self.V(x)
        a = self.A(x)
        x = v + a - a.mean(dim=-1, keepdim=True)
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