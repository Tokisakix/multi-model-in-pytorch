from torch.utils.data import DataLoader

from load_config import load_config
from data import get_dataset

def get_dataloader(CONFIG, dataset, mod):
    CONFIG = load_config()
    LOADER_CONFIG = CONFIG["dataloader"]

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=LOADER_CONFIG["train_batch_size"] if mod == "train" else LOADER_CONFIG["test_batch_size"],
    )

    return dataloader




# ---Test---

if __name__ == "__main__":
    CONFIG = load_config()
    
    train_dataset = get_dataset(CONFIG, mod="train")
    test_dataset  = get_dataset(CONFIG, mod="test")

    train_dataloader = get_dataloader(CONFIG, train_dataset, mod="train")
    test_dataloader  = get_dataloader(CONFIG, test_dataset,  mod="test")

    train_inputs, train_labels = train_dataloader.__iter__()._next_data()
    test_inputs,  test_labels  = test_dataloader.__iter__()._next_data()

    print(train_inputs.shape, train_labels.shape)
    print(test_inputs.shape, test_labels.shape)