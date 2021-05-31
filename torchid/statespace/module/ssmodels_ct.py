import torch
import torch.nn as nn
import numpy as np
from torch.jit import Final
from typing import List


class NeuralStateSpaceModel(nn.Module):
    r"""A state-space continuous-time model.

    Args:
        n_x (int): Number of state variables
        n_u (int): Number of input variables
        n_feat: (int, optional): Number of input features in the hidden layer. Default: 64
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True
        activation: (str): Activation function in the hidden layer. Either 'relu', 'softplus', 'tanh'. Default: 'relu'

    Examples::

        >>> ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)
    """

    n_x: Final[int]
    n_u: Final[int]
    n_feat: Final[int]

    def __init__(self, n_x, n_u, n_feat=64, scale_dx=1.0, init_small=True, activation='relu'):
        super(NeuralStateSpaceModel, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = n_feat
        self.scale_dx = scale_dx

        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'softplus':
            activation = nn.Softplus()
        elif activation == 'tanh':
            activation = nn.Tanh()

        self.net = nn.Sequential(
            nn.Linear(n_x+n_u, n_feat),  # 2 states, 1 input
            activation,
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


class MechanicalStateSpaceSystem(nn.Module):
    r"""A state-space continuous-time model for a 1-DoF mechanical system.
    The state-space model has two states (nx=2) and one input (nu=1). The states x0 and x1 correspond to position and
    velocity, respectively. The input u correspond to an input force.
    The derivative of position x0 is velocity x2, while the derivative of velocity is the output of a neural network
    with features u, x0, x1.

    Args:
        n_feat: (int, optional): Number of input features in the hidden layer. Default: 64
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True
        activation: (str): Activation function in the hidden layer. Either 'relu', 'softplus', 'tanh'. Default: 'relu'

    """

    n_x: Final[int]
    n_u: Final[int]
    n_feat: Final[int]

    def __init__(self, n_feat=64, init_small=True, typical_ts=1.0):
        super(MechanicalStateSpaceSystem, self).__init__()
        self.n_feat = n_feat
        self.typical_ts = typical_ts

        self.net = nn.Sequential(
            nn.Linear(3, n_feat),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(n_feat, 1)
        )

        # Small initialization is better for multi-step methods
        if init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-3)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, in_x, in_u):
        list_dx: List[torch.Tensor]
        in_xu = torch.cat((in_x, in_u), -1)  # concatenate x and u over the last dimension to create the [xu] input
        dx_v = self.net(in_xu)/self.typical_ts  # \dot x = f([xu])

        list_dx = [in_x[..., [1]], dx_v]
        dx = torch.cat(list_dx, -1)  # dot x = v, dot v = net
        return dx


class StateSpaceModelLin(nn.Module):
    r"""A state-space continuous-time model corresponding to the sum of a linear state-space model plus a non-linear
    part modeled as a neural network

    Args:
        A: (np.array): A matrix of the linear part of the model
        B: (np.array): B matrix of the linear part of the model

    """
    def __init__(self, A, B):
        super(StateSpaceModelLin, self).__init__()

        self.A = nn.Linear(2, 2, bias=False)
        self.A.weight = torch.nn.Parameter(torch.tensor(A.astype(np.float32)), requires_grad=False)
        self.B = nn.Linear(1, 2, bias=False)
        self.B.weight = torch.nn.Parameter(torch.tensor(B.astype(np.float32)), requires_grad=False)

    def forward(self, X, U):
        dx = self.A(X) + self.B(U)
        return dx   


class CascadedTanksNeuralStateSpaceModel(nn.Module):
    r"""A state-space model to represent the cascaded two-tank system.


    Args:
        n_feat: (int, optional): Number of input features in the hidden layer. Default: 0
        scale_dx: (str): Scaling factor for the neural network output. Default: 1.0
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True

    """

    def __init__(self, n_feat=64, scale_dx=1.0, init_small=True):
        super(CascadedTanksNeuralStateSpaceModel, self).__init__()
        self.n_feat = n_feat
        self.scale_dx = scale_dx

        # Neural network for the first state equation = NN(x_1, u)
        self.net_dx1 = nn.Sequential(
            nn.Linear(2, n_feat),
            nn.Tanh(),
            nn.Linear(n_feat, 1),
        )

        # Neural network for the first state equation = NN(x_1, x2)
        self.net_dx2 = nn.Sequential(
            nn.Linear(2, n_feat),
            nn.Tanh(),
            nn.Linear(n_feat, 1),
        )

        # Small initialization is better for multi-step methods
        if init_small:
            for m in self.net_dx1.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

        # Small initialization is better for multi-step methods
        if init_small:
            for m in self.net_dx2.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, in_x, in_u):

        # the first state derivative is NN(x1, u)
        in_1 = torch.cat((in_x[..., [0]], in_u), -1)  # concatenate 1st state component with input
        dx_1 = self.net_dx1(in_1)

        # the second state derivative is NN(x1, x2)
        in_2 = in_x
        dx_2 = self.net_dx2(in_2)

        # the state derivative is built by concatenation of dx_1 and dx_2, possibly scaled for numerical convenience
        dx = torch.cat((dx_1, dx_2), -1)
        dx = dx * self.scale_dx
        return dx


class CascadedTanksOverflowNeuralStateSpaceModel(nn.Module):
    r"""A state-space model to represent the cascaded two-tank system, with possible overflow from the lower tank.


    Args:
        n_feat: (int, optional): Number of input features in the hidden layer. Default: 0
        scale_dx: (str): Scaling factor for the neural network output. Default: 1.0
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True

    """

    def __init__(self, n_feat=64, scale_dx=1.0, init_small=True):
        super(CascadedTanksOverflowNeuralStateSpaceModel, self).__init__()
        self.n_feat = n_feat
        self.scale_dx = scale_dx

        # Neural network for the first state equation = NN(x_1, u)
        self.net_dx1 = nn.Sequential(
            nn.Linear(2, n_feat),
            nn.ReLU(),
            nn.Linear(n_feat, 1),
        )

        # Neural network for the first state equation = NN(x_1, x2, u)
        # we assume that with overflow the input may influence the 2nd tank instantaneously
        self.net_dx2 = nn.Sequential(
            nn.Linear(3, n_feat),
            nn.ReLU(),
            nn.Linear(n_feat, 1),
        )

        # Small initialization is better for multi-step methods
        if init_small:
            for m in self.net_dx1.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

        # Small initialization is better for multi-step methods
        if init_small:
            for m in self.net_dx2.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, in_x, in_u):

        # the first state derivative is NN_1(x1, u)
        in_1 = torch.cat((in_x[..., [0]], in_u), -1)  # concatenate 1st state component with input
        dx_1 = self.net_dx1(in_1)

        # the second state derivative is NN_2(x1, x2, u)
        in_2 = torch.cat((in_x, in_u), -1) # concatenate states with input to define the
        dx_2 = self.net_dx2(in_2)

        # the state derivative is built by concatenation of dx_1 and dx_2, possibly scaled for numerical convenience
        dx = torch.cat((dx_1, dx_2), -1)
        dx = dx * self.scale_dx
        return dx
