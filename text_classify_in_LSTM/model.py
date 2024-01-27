import torch
import torch.nn as nn
import torch.nn.functional as F

from load_config import load_config

class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb    = nn.Embedding(num_embeddings=vocab_size, embedding_dim=300)
        self.lstm   = nn.LSTM(input_size=300, hidden_size=128, num_layers=4, batch_first=True)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 4, out_features=84),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(in_features=84, out_features=2),
            nn.Softmax(dim=-1),
        )
        return
    
    def forward(self, inputs):
        outputs = self.emb(inputs)
        _, (outputs, _) = self.lstm(outputs)
        outputs = outputs.transpose(0, 1)
        outputs = self.linear(outputs)
        return outputs


# ---TEST---

if __name__ == "__main__":
    CONFIG  = load_config()
    CUDA    = CONFIG["cuda"]

    inputs  = torch.randint(0, 176694, (32, 128))
    model   = Model()

    inputs  = inputs.cuda() if CUDA else inputs
    model   = model.cuda() if CUDA else model
    outputs = model(inputs)

    print(inputs.shape)
    print(model.eval())
    print(outputs.shape)