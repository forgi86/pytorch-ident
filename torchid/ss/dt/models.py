import torch
import torch.nn as nn
from torchid.ss.poly_utils import valid_coeffs


class NeuralStateUpdate(nn.Module):
    r"""State-update mapping modeled as a feed-forward neural network with one hidden layer.

    The model has structure:

    .. math::
        \begin{aligned}
            x_{k+1} = x_k + \mathcal{N}(x_k, u_k),
        \end{aligned}

    where :math:`\mathcal{N}(\cdot, \cdot)` is a feed-forward neural network with one hidden layer.

    Args:
        n_x (int): Number of state variables
        n_u (int): Number of input variables
        hidden_size: (int, optional): Number of input features in the hidden layer. Default: 0
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True
        activation: (str): Activation function in the hidden layer. Either 'relu', 'softplus', 'tanh'. Default: 'relu'

    Examples::

        >>> ss_model = NeuralStateUpdate(n_x=2, n_u=1, hidden_size=64)
    """

    def __init__(self, n_x, n_u, hidden_size=16, init_small=True):
        super(NeuralStateUpdate, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = hidden_size
        self.net = nn.Sequential(
            nn.Linear(n_x + n_u, hidden_size),  # 2 states, 1 input
            nn.Tanh(),
            nn.Linear(hidden_size, n_x)
        )

        if init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x, u):
        xu = torch.cat((x, u), -1)
        dx = self.net(xu)
        return dx


class PolynomialStateUpdate(nn.Module):
    r"""State-update mapping modeled as a polynomial in x and u.

    The model has structure:

    .. math::
        \begin{aligned}
            x_{k+1} = x_k + Ax_{k} + Bu_{k} + Ez_{k},
        \end{aligned}

    where z_{k} is a vector containing (non-linear) monomials in x_{k} and u_{k}

    Args:
        n_x: (np.array): Number of states.
        n_u: (np.array): Number of inputs.
        d_max (int): Maximum degree of the polynomial model.

    """

    def __init__(self, n_x, n_u, d_max, init_small=True):
        super(PolynomialStateUpdate, self).__init__()

        self.n_x = n_x
        self.n_u = n_u
        poly_coeffs = valid_coeffs(n_x + n_u, d_max)
        self.n_poly = len(poly_coeffs)
        self.poly_coeffs = torch.tensor(poly_coeffs)
        self.A = nn.Linear(n_x, n_x, bias=False)
        self.B = nn.Linear(n_u, n_x, bias=False)
        # self.D = nn.linear(n_u, n_y, bias=False)
        self.E = nn.Linear(self.n_poly, n_x, bias=False)
        # self.F = nn.linear(self.n_poly, n_y)
        self.nl_on = True

        if init_small:
            nn.init.normal_(self.A.weight, mean=0, std=1e-3)
            nn.init.normal_(self.B.weight, mean=0, std=1e-3)
            nn.init.normal_(self.E.weight, mean=0, std=1e-6)

            # nn.init.constant_(module.bias, val=0)

    def enable_nl(self):
        self.nl_on = True

    def disable_nl(self):
        self.nl_on = False

    def freeze_nl(self):
        self.E.requires_grad_(False)

    def unfreeze_nl(self):
        self.E.requires_grad_(True)

    def freeze_lin(self):
        self.A.requires_grad_(False)
        self.B.requires_grad_(False)

    def unfreeze_lin(self):
        self.A.requires_grad_(True)
        self.B.requires_grad_(True)

    def forward(self, x, u):
        xu = torch.cat((x, u), dim=-1)
        xu_ = xu.unsqueeze(xu.ndim - 1)

        dx = self.A(x) + self.B(u)
        if self.nl_on:
            zeta = torch.prod(torch.pow(xu_, self.poly_coeffs), axis=-1)
            # eta = torch.prod(torch.pow(xu_, self.poly_coeffs), axis=-1)
            dx = dx + self.E(zeta)
        return dx


class NeuralLinStateUpdate(nn.Module):
    r"""State-update mapping modeled as a feed-forward neural network with one hidden layer.

    The model has structure:

    .. math::
        \begin{aligned}
            x_{k+1} = x_k + \mathcal{N}(x_k, u_k),
        \end{aligned}

    where :math:`\mathcal{N}(\cdot, \cdot)` is a feed-forward neural network with one hidden layer.

    Args:
        n_x (int): Number of state variables
        n_u (int): Number of input variables
        hidden_size: (int, optional): Number of input features in the hidden layer. Default: 0
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True
        activation: (str): Activation function in the hidden layer. Either 'relu', 'softplus', 'tanh'. Default: 'relu'

    Examples::

        >>> ss_model = NeuralStateUpdate(n_x=2, n_u=1, hidden_size=64)
    """

    def __init__(self, n_x, n_u, hidden_size=16, init_small=True):
        super(NeuralLinStateUpdate, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(n_x + n_u, hidden_size),  # 2 states, 1 input
            nn.Tanh(),
            nn.Linear(hidden_size, n_x)
        )
        self.lin = nn.Linear(n_x + n_u, n_x, bias=False)
        self.nl_on = True

        if init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

            for m in self.lin.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)

    def freeze_nl(self):
        self.net.requires_grad_(False)

    def unfreeze_nl(self):
        self.net.requires_grad_(True)

    def freeze_lin(self):
        self.lin.requires_grad_(False)

    def unfreeze_lin(self):
        self.lin.requires_grad_(True)

    def enable_nl(self):
        self.nl_on = True

    def disable_nl(self):
        self.nl_on = False

    def forward(self, x, u):
        xu = torch.cat((x, u), -1)
        dx = self.lin(xu)
        if self.nl_on:
            dx = dx + self.net(xu)
        return dx


class LinearStateUpdate(nn.Module):
    r"""State-update mapping modeled as a linear function in x and u.

    The model has structure:

    .. math::
        \begin{aligned}
            x_{k+1} = x_k + Ax_{k} + Bu_{k}.
        \end{aligned}

    Args:
        n_x: (np.array): Number of states.
        n_u: (np.array): Number of inputs.
        d_max (int): Maximum degree of the polynomial model.

    """

    def __init__(self, n_x, n_u, init_small=True):
        super(LinearStateUpdate, self).__init__()

        self.n_x = n_x
        self.n_u = n_u
        self.A = nn.Linear(n_x, n_x, bias=False)
        self.B = nn.Linear(n_u, n_x, bias=False)

        if init_small:
            for module in [self.A, self.B]:
                nn.init.normal_(module.weight, mean=0, std=1e-2)
                # nn.init.constant_(module.bias, val=0)

    def forward(self, x, u):
        dx = self.A(x) + self.B(u)
        return dx


class CTSNeuralStateSpace(nn.Module):
    r"""A state-space model to represent the cascaded two-tank system.

    Args:
        hidden_size: (int, optional): Number of input features in the hidden layer. Default: 0
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True
    """

    def __init__(self, n_x, n_u, hidden_size=64, init_small=True):
        super(CTSNeuralStateSpace, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = hidden_size
        self.net = nn.Sequential(
            nn.Linear(n_x + n_u, hidden_size),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(hidden_size, n_x)
        )

        if init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x, u):
        xu = torch.cat((x, u), -1)
        dx = self.net(xu)
        return dx


class LinearOutput(nn.Module):
    r"""Output  mapping modeled as a linear function in x.

    The model has structure:

    .. math::
        \begin{aligned}
            y_{k} = Cx_k.
        \end{aligned}
    """

    def __init__(self, n_x, n_y, bias=False):
        super(LinearOutput, self).__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.C = torch.nn.Linear(n_x, n_y, bias=bias)

    def forward(self, x):
        return self.C(x)


class NeuralOutput(nn.Module):
    r"""Output  mapping modeled as a feed-forward neural network in x.

    The model has structure:

    .. math::
        \begin{aligned}
            y_{k} = \mathcal{N}(x_k).
        \end{aligned}
    """

    def __init__(self, n_x, n_y, hidden_size=16):
        super(NeuralOutput, self).__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.net = nn.Sequential(nn.Linear(n_x, hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_size, n_y)
                                 )

    def forward(self, x):
        return self.net(x)


class NeuralLinOutput(nn.Module):
    r"""Output  mapping modeled as a feed-forward neural network in x.

    The model has structure:

    .. math::
        \begin{aligned}
            y_{k} = \mathcal{N}(x_k).
        \end{aligned}
    """

    def __init__(self, n_x, n_y, hidden_size=16):
        super(NeuralLinOutput, self).__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.net = nn.Sequential(nn.Linear(n_x, hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_size, n_y)
                                 )

        self.lin = nn.Linear(n_x, n_y, bias=False)
        self.nl_on = True

    def freeze_nl(self):
        self.net.requires_grad_(False)

    def unfreeze_nl(self):
        self.net.requires_grad_(True)

    def freeze_lin(self):
        self.lin.requires_grad_(False)

    def unfreeze_lin(self):
        self.lin.requires_grad_(True)

    def enable_nl(self):
        self.nl_on = True

    def disable_nl(self):
        self.nl_on = False

    def forward(self, x):
        y = self.lin(x)
        if self.nl_on:
            y += self.net(x)
        return y


class ChannelsOutput(nn.Module):
    r"""Output  mapping corresponding to a specific state channel.

    """

    def __init__(self, channels):
        super(ChannelsOutput, self).__init__()
        self.channels = channels

    def forward(self, x):
        y = x[..., self.channels]
        return y
