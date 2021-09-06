import torch
import torch.nn as nn
import numpy as np


class NeuralIOModel(nn.Module):
    """ This class implements an IO neural model

     Attributes
     ----------
     n_a : int.
           number of autoregressive lags in y
     n_b : int.
           number of autoregressive lags in u
     n_feat : int.
           number of units in the hidden layer
     """
    def __init__(self, n_a, n_b, n_feat=64, small_init=True):
        super(NeuralIOModel, self).__init__()
        self.n_a = n_a
        self.n_b = n_b
        self.n_feat = n_feat

        const_np = np.zeros((n_a + n_b, 1), dtype=np.float32)
        const_np[0, 0] = 1.0
        self.const = torch.tensor(const_np)

        self.net = nn.Sequential(
            nn.Linear(n_a + n_b, n_feat),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(n_feat, 1),
        )

        if small_init:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, phi):
        Y = self.net(phi) + torch.matmul(phi, self.const)
        return Y


class NeuralIOModelComplex(nn.Module):
    def __init__(self, n_a, n_b, n_feat=64, small_init=True):
        super(NeuralIOModelComplex, self).__init__()
        self.n_a = n_a
        self.n_b = n_b
        self.n_feat = 64

        const_np = np.zeros((n_a + n_b, 1), dtype=np.float32)
        const_np[0, 0] = 1.0
        self.const = torch.tensor(const_np)

        self.net = nn.Sequential(
            nn.Linear(n_a + n_b, n_feat),  # 2 states, 1 input
            nn.ELU(),
            nn.Linear(n_feat, n_feat),
            nn.ELU(),
            nn.Linear(n_feat, 1)
        )

        if small_init:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-3)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, phi):
        Y = self.net(phi) + torch.matmul(phi, self.const)
        return Y
