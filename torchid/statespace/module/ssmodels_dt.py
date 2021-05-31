import torch
import torch.nn as nn
import numpy as np
from torch.jit import Final


class NeuralStateSpaceModel(nn.Module):

    r"""A state-space discrete-time model. The state mapping is a neural network with one hidden layer.

    Args:
        n_x (int): Number of state variables
        n_u (int): Number of input variables
        n_feat: (int, optional): Number of input features in the hidden layer. Default: 0
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True
        activation: (str): Activation function in the hidden layer. Either 'relu', 'softplus', 'tanh'. Default: 'relu'

    Examples::

        >>> ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)
    """

    def __init__(self, n_x, n_u, n_feat=64, init_small=True):
        super(NeuralStateSpaceModel, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = n_feat
        self.net = nn.Sequential(
            nn.Linear(n_x+n_u, n_feat),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(n_feat, n_x)
        )

        if init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)
    
    def forward(self, X, U):
        XU = torch.cat((X, U), -1)
        DX = self.net(XU)
        return DX


class DeepNeuralStateSpaceModel(nn.Module):
    r"""A state-space discrete-time model. The state mapping is a neural network with two hidden layers.


    Args:
        n_x (int): Number of state variables
        n_u (int): Number of input variables
        n_feat: (int, optional): Number of input features in the two hidden layer. Default:64
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True
        activation: (str): Activation function in the hidden layer. Either 'relu', 'softplus', 'tanh'. Default: 'relu'

    Examples::

        >>> ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)
    """

    n_x: Final[int]
    n_u: Final[int]
    n_feat: Final[int]

    def __init__(self, n_x, n_u, n_feat=64, scale_dx=1.0, init_small=True):
        super(DeepNeuralStateSpaceModel, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = n_feat
        self.scale_dx = scale_dx

        self.net = nn.Sequential(
            nn.Linear(n_x + n_u, n_feat),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(n_feat, n_feat),
            nn.ReLU(),
            nn.Linear(n_feat, n_x)
        )

        # Small initialization is better for multi-step methods
        if init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, in_x, in_u):
        in_xu = torch.cat((in_x, in_u), -1)  # concatenate x and u over the last dimension to create the [xu] input
        dx = self.net(in_xu)  # \dot x = f([xu])
        dx = dx * self.scale_dx
        return dx


class StateSpaceModelLin(nn.Module):
    r"""A state-space continuous-time model corresponding to the sum of a linear state-space model plus a non-linear
    part modeled as a neural network

    Args:
        A: (np.array): A matrix of the linear part of the model
        B: (np.array): B matrix of the linear part of the model

    """

    def __init__(self, AL, BL):
        super(StateSpaceModelLin, self).__init__()

        self.AL = nn.Linear(2, 2, bias=False)
        self.AL.weight = torch.nn.Parameter(torch.tensor(AL.astype(np.float32)), requires_grad=False)
        self.BL = nn.Linear(1, 2, bias=False)
        self.BL.weight = torch.nn.Parameter(torch.tensor(BL.astype(np.float32)), requires_grad=False)
    
    def forward(self, X, U):
        DX = self.AL(X) + self.BL(U)
        return DX   


class CTSNeuralStateSpaceModel(nn.Module):
    r"""A state-space model to represent the cascaded two-tank system.


    Args:
        n_feat: (int, optional): Number of input features in the hidden layer. Default: 0
        scale_dx: (str): Scaling factor for the neural network output. Default: 1.0
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True

    """

    def __init__(self, n_x, n_u, n_feat=64, ts=1.0, init_small=True):
        super(CTSNeuralStateSpaceModel, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = n_feat
        self.ts = ts
        self.net = nn.Sequential(
            nn.Linear(n_x + n_u, n_feat),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(n_feat, n_x)
        )

        if init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, X, U):
        XU = torch.cat((X, U), -1)
        DX = self.net(XU) * self.ts
        return DX