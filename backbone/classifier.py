import torch.nn as nn
import torch.nn.functional as F
import torch


class Classifier(nn.Module):
    def __init__(self, nhid, nclass):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, x):
        x = self.fc(x)
        return x