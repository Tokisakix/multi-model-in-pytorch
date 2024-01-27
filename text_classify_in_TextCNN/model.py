import torch
import torch.nn as nn
import torch.nn.functional as F

from load_config import load_config

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb   = nn.Embedding(num_embeddings=176694, embedding_dim=300)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, stride=7)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=11, stride=11)
        self.linear = nn.Sequential(
            nn.Linear(in_features=128 * 3, out_features=84),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(in_features=84, out_features=2),
            nn.Softmax(dim=-1),
        )
        return
    
    def forward(self, inputs):
        outputs  = self.emb(inputs)
        outputs1 = self.conv1(outputs)
        outputs2 = self.conv2(outputs)
        outputs3 = self.conv3(outputs)

        outputs1 = F.max_pool1d(outputs1, kernel_size=outputs1.shape[-1], stride=outputs1.shape[-1])
        outputs2 = F.max_pool1d(outputs2, kernel_size=outputs2.shape[-1], stride=outputs2.shape[-1])
        outputs3 = F.max_pool1d(outputs3, kernel_size=outputs3.shape[-1], stride=outputs3.shape[-1])

        outputs = torch.cat((outputs1.squeeze(-1), outputs2.squeeze(-1), outputs3.squeeze(-1)), dim=-1)
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