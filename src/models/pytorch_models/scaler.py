# From https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients/blob/master/src/modules/modules.py
import torch.nn as nn


class Scaler(nn.Module):
    def __init__(self, rate, keep_scaling=False):
        super().__init__()
        self.rate = rate
        self.keep_scaling = keep_scaling

    def forward(self, input):
        scale = self.keep_scaling or self.training
        output = input / self.rate if scale else input
        return output
