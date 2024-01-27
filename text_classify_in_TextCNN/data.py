import torch
from torch.utils.data import TensorDataset
from torchtext import datasets, vocab

from load_config import load_config

def get_raw_data(CONFIG, mod):
    DATA_CONFIG = CONFIG["data"]

    raw_data_set = datasets.IMDB(DATA_CONFIG["root"], split=mod)
    raw_text_set  = [[word for word in item[1].split(" ")] for item in iter(raw_data_set)]
    raw_label_set = [item[0] - 1 for item in iter(raw_data_set)]

    return raw_text_set, raw_label_set

def get_data(raw_text_set, raw_label_set, _word2idx=None):
    text_set  = []
    label_set = torch.tensor(raw_label_set)
    word2idx  = {} if _word2idx == None else _word2idx
    idx = 0

    for raw_sentence in raw_text_set:
        sentence = []
        for word in raw_sentence:
            if not word in word2idx:
                if _word2idx == None:
                    idx += 1
                    word2idx[word] = idx
                else:
                    word2idx[word] = 0
            sentence.append(word2idx[word])
        sentence = torch.tensor(sentence)
        text_set.append(sentence)

    return text_set, label_set, word2idx, idx

def get_dataset(text_set, label_set):
    text_set = torch.nn.utils.rnn.pad_sequence(text_set, True)[:, :128]
    dataset = TensorDataset(text_set.long(), label_set.long())
    return dataset



# ---Test---
    
if __name__ == "__main__":
    CONFIG = load_config()
    
    raw_train_text_set, raw_train_label_set = get_raw_data(CONFIG, "train")
    raw_test_text_set,  raw_test_label_set  = get_raw_data(CONFIG, "test")

    train_text_set, train_label_set, word2idx, vocab_size = get_data(raw_train_text_set, raw_train_label_set)
    test_text_set,  test_label_set, _, _  = get_data(raw_test_text_set,  raw_test_label_set, word2idx)

    train_dataset = get_dataset(train_text_set, train_label_set)
    test_dataset  = get_dataset(test_text_set,  test_label_set)