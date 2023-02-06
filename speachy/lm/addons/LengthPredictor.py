import torch, torch.nn as nn, torch.nn.functional as F


class LengthPredictor(nn.Module):
    '''Takes the length of a sentence in seconds and predicts a vector that is added to the bos token embedding'''
    def __init__(self, dim):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.mlp(x)
    