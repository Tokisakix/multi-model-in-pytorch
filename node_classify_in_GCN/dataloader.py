from torch.utils.data import DataLoader

from load_config import load_config
from data import get_raw_data, get_data, get_dataset

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
    
    raw_train_text_set, raw_train_label_set = get_raw_data(CONFIG, "train")
    raw_test_text_set,  raw_test_label_set  = get_raw_data(CONFIG, "test")

    train_text_set, train_label_set, word2idx, vocab_size = get_data(raw_train_text_set, raw_train_label_set)
    test_text_set,  test_label_set, _, _  = get_data(raw_test_text_set,  raw_test_label_set, word2idx)

    train_dataset = get_dataset(train_text_set, train_label_set)
    test_dataset  = get_dataset(test_text_set,  test_label_set)

    train_dataloader = get_dataloader(CONFIG, train_dataset, mod="train")
    test_dataloader  = get_dataloader(CONFIG, test_dataset,  mod="test")

    train_inputs, train_labels = train_dataloader.__iter__()._next_data()
    test_inputs,  test_labels  = test_dataloader.__iter__()._next_data()

    print(train_inputs, train_labels)
    print(test_inputs, test_labels)