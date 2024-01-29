from torch.utils.data import DataLoader

from load_config import load_config
from data import get_dataset

def get_dataloader(CONFIG, dataset):
    LOADER_CONFIG = CONFIG["dataloader"]

    inputs = dataset.x
    edges  = dataset.edge_index
    labels = dataset.y

    return inputs, edges, labels




# ---Test---

if __name__ == "__main__":
    CONFIG = load_config()

    dataset = get_dataset(CONFIG)

    inputs, edges, labels = get_dataloader(CONFIG, dataset)
    
    print(inputs.shape, edges.shape, labels.shape)