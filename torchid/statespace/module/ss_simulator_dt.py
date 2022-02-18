import torch
import torch.nn as nn
from typing import List

        
class NeuralStateSpaceSimulator(object):
    r""" Discrete-time state-space simulator.

    Args:
        ss_model (nn.Module): The neural state-space model.
        batch_first (bool): If True, first dimension is batch.

    Inputs: x_0, input
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
        >>> nn_solution = NeuralStateSpaceSimulator(ss_model)
        >>> x0 = torch.randn(64, 3)
        >>> u = torch.randn(100, 64, 2)
        >>> x = nn_solution(x0, u)
        >>> print(x.size())
        torch.Size([100, 64, 3])
     """

    def __init__(self, ss_model):
        self.ss_model = ss_model

    def forward(self, x_0, input):

        x: List[torch.Tensor] = []
        x_step = x_0
        dim_time = 1 if self.batch_first else 0

        for u_step in input.split(1, dim=dim_time):  # split along the time axis
            u_step = u_step.squeeze(dim_time)
            x += [x_step]
            dx = self.ss_model(x_step, u_step)
            x_step = x_step + dx

        x = torch.stack(x, dim_time)
        return x
