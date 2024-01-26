import torch
import torch.nn as nn

from load_config import load_config

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        return

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = outputs.flatten(2).transpose(1, 2)
        return outputs

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = PatchEmbedding(patch_size=8, in_channels=3, embed_dim=192)
        self.pos_embedding = nn.Parameter(torch.rand(1, 16, 192))
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=192,
                nhead=4,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=192, out_features=1024),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=10),
            nn.Softmax(dim=-1),
        )
        return
    
    def forward(self, inputs):
        outputs = self.embedding(inputs)
        outputs += self.pos_embedding
        outputs = self.transformer(outputs)
        outputs = outputs.mean(dim=1)
        outputs = self.linear(outputs)
        return outputs


# ---TEST---

if __name__ == "__main__":
    CONFIG  = load_config()
    CUDA    = CONFIG["cuda"]

    inputs  = torch.randn(32, 3, 32, 32)
    model   = Model()

    inputs  = inputs.cuda() if CUDA else inputs
    model   = model.cuda() if CUDA else model
    outputs = model(inputs)

    print(inputs.shape)
    print(model.eval())
    print(outputs.shape)