import torch, torch.nn as nn, torch.nn.functional as F


class NextSentenceTokenAdapter(nn.Module):
    '''Takes logits from the last token of the sequence and applies a transformation to it to get the logits for the first token of the next sequence
        This is to avoid bos token, mainly the transformation here should be downweighting the eos token and upweighting certain tokens like 'the' or 'a' etc.
    '''
    def __init__(self, dim):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.randn(1, dim))
        self.beta = nn.Parameter(torch.randn(1, dim))
        torch.nn.init.normal_(self.gamma, std=0.02)
        torch.nn.init.normal_(self.beta, std=0.02)

    def forward(self, x):
        return x * self.gamma + self.beta
    