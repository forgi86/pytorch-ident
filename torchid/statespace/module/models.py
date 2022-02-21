import torch
import torch.nn as nn
from torchid.statespace.module.poly_utils import valid_coeffs


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


class PolynomialStateSpaceModel(nn.Module):
    r"""A state-space continuous-time model corresponding to the sum of a linear state-space model plus a non-linear
    part modeled as a neural network

    Args:
        n_x: (np.array): Number of states.
        n_u: (np.array): Number of inputs.
        d_max (int): Maximum degree of the polynomial model.

    """

    def __init__(self, n_x, n_u, d_max):
        super(PolynomialStateSpaceModel, self).__init__()

        poly_coeffs = valid_coeffs(n_x + n_u, d_max)
        self.n_poly = len(poly_coeffs)
        self.poly_coeffs = torch.tensor(poly_coeffs)
        self.A = nn.Linear(n_x, n_x, bias=False)
        self.B = nn.Linear(n_u, n_x, bias=False)
        #self.D = torch.randn(n_y, n_u)
        self.E = torch.randn(n_x, self.n_poly)
        #self.F = torch.randn(n_y, self.n_poly)

    def forward(self, x, u):
        xu = torch.cat((x, u), dim=-1)
        xu_ = xu.unsqueeze(xu.ndim - 1)
        zeta = torch.prod(torch.pow(xu_, self.poly_coeffs), axis=-1)
        #eta = torch.prod(torch.pow(xu_, self.poly_coeffs), axis=-1)

        dx = self.A(x) + self.B(u) + self.E(zeta)
        return dx


class CTSNeuralStateSpaceModel(nn.Module):
    r"""A state-space model to represent the cascaded two-tank system.
    Args:
        n_feat: (int, optional): Number of input features in the hidden layer. Default: 0
        scale_dx: (str): Scaling factor for the neural network output. Default: 1.0
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True
    """

    def __init__(self, n_x, n_u, n_feat=64, init_small=True):
        super(CTSNeuralStateSpaceModel, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = n_feat
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
        DX = self.net(XU)
        return DX