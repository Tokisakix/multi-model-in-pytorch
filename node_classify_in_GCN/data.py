import torch
from torch_geometric import torch_geometric

from load_config import load_config

def get_dataset(CONFIG):
    dataset = torch_geometric.datasets.CoraFull(CONFIG["data"]["root"])[0]
    return dataset



# ---Test---
    
if __name__ == "__main__":
    CONFIG = load_config()

    dataset = get_dataset(CONFIG)
    for graph in dataset:
        print(graph)