import torch
from torch_geometric import torch_geometric
import torch.nn.functional as F

from load_config import load_config

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = torch_geometric.nn.GCNConv(in_channels=8710, out_channels=128)
        self.gcn2 = torch_geometric.nn.GCNConv(in_channels=128, out_channels=70)
        return
    
    def forward(self, inputs, edges):
        outputs = self.gcn1(inputs, edges)
        outputs = F.relu(outputs)
        outputs = self.gcn2(outputs, edges)
        outputs = F.softmax(outputs, dim=-1)
        return outputs


# ---TEST---

if __name__ == "__main__":
    CONFIG  = load_config()
    CUDA    = CONFIG["cuda"]

    inputs  = torch.randn(19793, 8710)
    edges   = torch.randint(0, 1, (2, 126842))
    model   = Model()

    inputs  = inputs.cuda() if CUDA else inputs
    edges   = edges.cuda() if CUDA else edges
    model   = model.cuda() if CUDA else model
    outputs = model(inputs, edges)

    print(inputs.shape)
    print(edges.shape)
    print(model.eval())
    print(outputs.shape)