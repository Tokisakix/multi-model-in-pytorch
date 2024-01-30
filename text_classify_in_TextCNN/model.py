import torch
import torch.nn as nn
import torch.nn.functional as F

from load_config import load_config

class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb   = nn.Embedding(num_embeddings=vocab_size, embedding_dim=300)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 300), stride=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(7, 300), stride=1)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(11, 300), stride=1)
        self.linear = nn.Sequential(
            nn.Linear(in_features=128 * 3, out_features=84),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(in_features=84, out_features=2),
            nn.Softmax(dim=-1),
        )
        return
    
    def forward(self, inputs):
        outputs  = self.emb(inputs).unsqueeze(1)
        outputs1 = self.conv1(outputs)
        outputs2 = self.conv2(outputs)
        outputs3 = self.conv3(outputs)

        outputs1 = F.max_pool2d(outputs1, kernel_size=(outputs1.shape[-2], 1), stride=1).reshape(-1, 128)
        outputs2 = F.max_pool2d(outputs2, kernel_size=(outputs2.shape[-2], 1), stride=1).reshape(-1, 128)
        outputs3 = F.max_pool2d(outputs3, kernel_size=(outputs3.shape[-2], 1), stride=1).reshape(-1, 128)

        outputs = torch.cat((outputs1, outputs2, outputs3), dim=-1)
        outputs = self.linear(outputs)

        return outputs


# ---TEST---

if __name__ == "__main__":
    CONFIG  = load_config()
    CUDA    = False

    inputs  = torch.randint(0, 176694, (32, 128))
    model   = Model(300_000)

    inputs  = inputs.cuda() if CUDA else inputs
    model   = model.cuda() if CUDA else model
    outputs = model(inputs)

    print(inputs.shape)
    print(model.eval())
    print(outputs.shape)