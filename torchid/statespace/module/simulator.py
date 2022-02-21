import torch
import torch.nn as nn
from typing import List

        
class StateSpaceSimulator(nn.Module):
    r""" Discrete-time state-space simulator.

    Args:
        f_xu (nn.Module): The neural state-space model.
        batch_first (bool): If True, first dimension is batch.

    Inputs: x_0, u
        * **x_0**: tensor of shape :math:`(N, n_{x})` containing the
          initial hidden state for each element in the batch.
          Defaults to zeros if (h_0, c_0) is not provided.
        * **input**: tensor of shape :math:`(L, N, n_{u})` when ``batch_first=False`` or
          :math:`(N, L, n_{x})` when ``batch_first=True`` containing the input sequence

    Outputs: x
        * **x**: tensor of shape :math:`(L, N, n_{x})` corresponding to
          the simulated state sequence.

    Examples::

        >>> ss_model = NeuralStateSpaceModel(n_x=3, n_u=2)
        >>> nn_solution = StateSpaceSimulator(ss_model)
        >>> x0 = torch.randn(64, 3)
        >>> u = torch.randn(100, 64, 2)
        >>> x = nn_solution(x0, u)
        >>> print(x.size())
        torch.Size([100, 64, 3])
     """

    def __init__(self, f_xu, g_xu=None, batch_first=False):
        super().__init__()
        self.state_update = f_xu
        self.output = g_xu
        self.batch_first = batch_first

    def simulate_state(self, x_0, u):
        x: List[torch.Tensor] = []
        x_step = x_0
        dim_time = 1 if self.batch_first else 0

        for u_step in u.split(1, dim=dim_time):  # split along the time axis
            u_step = u_step.squeeze(dim_time)
            x += [x_step]
            dx = self.state_update(x_step, u_step)
            x_step = x_step + dx

        x = torch.stack(x, dim_time)
        return x

    def forward(self, x_0, u, return_x=False):
        x = self.simulate_state(x_0, u)
        if self.output is not None:
            y = self.output(x)
        else:
            y = x
        if not return_x:
            return y
        else:
            return y, x


