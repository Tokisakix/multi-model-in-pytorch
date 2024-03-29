import torchvision
from torchvision import datasets
from torch.utils.data import Dataset

from load_config import load_config

class DataSet(Dataset):
    def __init__(self, root, download, mod):
        super().__init__()
        self.dataset = datasets.CIFAR10(
            root=root, 
            train=(mod=="train"), 
            download=download, 
            transform=torchvision.transforms.ToTensor()
        )
        return
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label
    
    def __len__(self):
        length = len(self.dataset)
        return length

def get_dataset(CONFIG, mod):
    DATA_CONFIG = CONFIG["data"]

    dataset = DataSet(
        root=DATA_CONFIG["root"],
        download=DATA_CONFIG["download"],
        mod=mod,
    )

    return dataset


# ---Test---
    
if __name__ == "__main__":
    CONFIG = load_config()
    
    train_dataset = get_dataset(CONFIG, "train")
    test_dataset  = get_dataset(CONFIG, "test")

    print(train_dataset, len(train_dataset))
    print(test_dataset,  len(test_dataset))