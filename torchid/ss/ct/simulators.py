import torch
import torch.nn as nn
import numpy as np
import nodepy
from typing import List


class ForwardEulerSimulator(nn.Module):

    r""" Forward Euler integration of a continuous-time neural state space model.

    Args:
        ss_model (nn.Module): The neural state-space model.
        ts (np.float): Sampling time for simulation.
        batch_first (bool): If True, first dimension is batch.

    Inputs: x_0, input
        * **x_0**: tensor of shape :math:`(N, n_{x})` containing the
          initial hidden state for each element in the batch.
          Defaults to zeros if (h_0, c_0) is not provided.
        * **input**: tensor of shape :math:`(L, N, n_{u})` when ``batch_first=False`` or
          :math:`(N, L, n_{x})` when ``batch_first=True`` containing the input sequence

    Outputs: states
        * **states**: tensor of shape :math:`(L, N, n_{x})` corresponding to
          the simulated state sequence.

    Examples::

        >>> ss_model = NeuralStateSpaceModel(n_x=3, n_u=2)
        >>> nn_solution = ForwardEulerSimulator(ss_model)
        >>> x0 = torch.randn(64, 3)
        >>> u = torch.randn(100, 64, 2)
        >>> x = nn_solution(x0, u)
        >>> print(x.size())
        torch.Size([100, 64, 3])
     """

    def __init__(self, ss_model, ts=1.0, batch_first=False):
        super(ForwardEulerSimulator, self).__init__()
        self.ss_model = ss_model
        self.ts = ts
        self.batch_first = batch_first

    def forward(self, x_0: torch.Tensor, input: torch.Tensor) -> torch.Tensor:

        states: List[torch.Tensor] = []
        x_step = x_0
        dim_time = 1 if self.batch_first else 0

        for u_step in input.split(1, dim=dim_time):  # split along the time axis
            u_step = u_step.squeeze(dim_time)
            states += [x_step]
            dx = self.ss_model(x_step, u_step)
            x_step = x_step + self.ts*dx

        states = torch.stack(states, dim_time)
        return states


class ExplicitRKSimulator(nn.Module):
    """ This class implements prediction/simulation methods for a continuous SS model structure

     Attributes
     ----------
     ss_model: nn.Module
               The neural SS model to be fitted
     ts: float
         model sampling time (when it is fixed)

     scheme: string
          Runge-Kutta scheme to be used
    """

    def __init__(self, ss_model, ts=1.0, scheme='RK44'):
        super(ExplicitRKSimulator, self).__init__()
        self.ss_model = ss_model
        self.ts = ts
        info_RK = nodepy.runge_kutta_method.loadRKM(scheme)
        self.A = torch.FloatTensor(info_RK.A.astype(np.float32))
        self.b = torch.FloatTensor(info_RK.b.astype(np.float32))
        self.c = torch.FloatTensor(info_RK.c.astype(np.float32))
        self.stages = self.b.numel()  # number of stages of the rk method

    def forward(self, x0_batch, u_batch):
        """ Multi-step simulation over (mini)batches

        Parameters
        ----------
        x0_batch: Tensor. Size: (q, n_x)
             Initial state for each subsequence in the minibatch

        u_batch: Tensor. Size: (m, q, n_u)
            Input sequence for each subsequence in the minibatch

        Returns
        -------
        Tensor. Size: (m, q, n_x)
            Simulated state for all subsequences in the minibatch

        """

        batch_size = x0_batch.shape[0]
        n_x = x0_batch.shape[1]
        seq_len = u_batch.shape[0]

        X_sim_list = []
        x_step = x0_batch
        for u_step in u_batch.split(1):#i in range(seq_len):

            u_step = u_step.squeeze(0)
            X_sim_list += [x_step]
            #u_step = u_batch[i, :, :]

            K = []  #torch.zeros((self.stages, nx))
            for stage_idx in range(self.stages):  # compute Ki, i=0,1,..s-1
                DX_pred = torch.zeros((batch_size, n_x)).to(self.device)
                for j in range(stage_idx):  # j=0,1,...i-1
                    DX_pred = DX_pred +  self.A[stage_idx, j] * K[j]
                DX_pred = DX_pred*self.ts
                K.append(self.ss_model(x_step + DX_pred, u_step))  # should u be interpolated??
            F = torch.zeros((batch_size, n_x)).to(self.device)
            for stage_idx in range(self.stages):
                F += self.b[stage_idx]*K[stage_idx]
            x_step = x_step + self.ts*F

        X_sim = torch.stack(X_sim_list, 0)

        return X_sim


class RK4Simulator(nn.Module):
    """ This class implements prediction/simulation methods for a continuous SS model structure

     Attributes
     ----------
     ss_model: nn.Module
               The neural SS model to be fitted
     ts: float
         model sampling time (when it is fixed)

     scheme: string
          Runge-Kutta scheme to be used
    """

    def __init__(self, ss_model, ts=1.0):
        super(RK4Simulator, self).__init__()
        self.ss_model = ss_model
        self.ts = ts

    def forward(self, x0_batch, u_batch):
        """ Multi-step simulation over (mini)batches

        Parameters
        ----------
        x0_batch: Tensor. Size: (q, n_x)
             Initial state for each subsequence in the minibatch

        u_batch: Tensor. Size: (m, q, n_u)
            Input sequence for each subsequence in the minibatch

        Returns
        -------
        Tensor. Size: (m, q, n_x)
            Simulated state for all subsequences in the minibatch

        """

        X_sim_list = []
        x_step = x0_batch
        for u_step in u_batch.split(1):

            u_step = u_step.squeeze(0)
            X_sim_list += [x_step]

            dt2 = self.ts / 2.0
            k1 = self.ss_model(x_step, u_step)
            k2 = self.ss_model(x_step + dt2 * k1, u_step)
            k3 = self.ss_model(x_step + dt2 * k2, u_step)
            k4 = self.ss_model(x_step + self.ts * k3, u_step)
            dx = self.ts / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            x_step = x_step + dx

        X_sim = torch.stack(X_sim_list, 0)

        return X_sim