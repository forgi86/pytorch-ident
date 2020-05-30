import torch
import torch.nn as nn
import numpy as np
import nodepy
from typing import List


class ForwardEulerSimulator(nn.Module):

    """ This class implements prediction/simulation methods for the SS model structure

     Attributes
     ----------
     ss_model: nn.Module
               The neural SS model to be fitted
     ts: float
         model sampling time

     """

    def __init__(self, ss_model, ts=1.0):
        super(ForwardEulerSimulator, self).__init__()
        self.ss_model = ss_model
        self.ts = ts

    def forward(self, x0_batch: torch.Tensor, u_batch: torch.Tensor) -> torch.Tensor:
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

        X_sim_list: List[torch.Tensor] = []
        x_step = x0_batch

        for u_step in u_batch.split(1):  # i in range(seq_len):
            u_step = u_step.squeeze(0)
            X_sim_list += [x_step]
            dx = self.ss_model(x_step, u_step)
            x_step = x_step + self.ts*dx

        X_sim = torch.stack(X_sim_list, 0)
        return X_sim


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

    def __init__(self, ss_model, ts=1.0, scheme='RK44', device="cpu"):
        super(ExplicitRKSimulator, self).__init__()
        self.ss_model = ss_model
        self.ts = ts
        info_RK = nodepy.runge_kutta_method.loadRKM(scheme)
        self.A = torch.FloatTensor(info_RK.A.astype(np.float32))
        self.b = torch.FloatTensor(info_RK.b.astype(np.float32))
        self.c = torch.FloatTensor(info_RK.c.astype(np.float32))
        self.stages = self.b.numel()  # number of stages of the rk method
        self.device = device

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

    def __init__(self, ss_model, ts=1.0, scheme='RK44', device="cpu"):
        super(RK4Simulator, self).__init__()
        self.ss_model = ss_model
        self.ts = ts
        self.device = device

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
        for u_step in u_batch.split(1):#i in range(seq_len):

            u_step = u_step.squeeze(0)
            X_sim_list += [x_step]
            #u_step = u_batch[i, :, :]

            dt2 = self.ts / 2.0
            k1 = self.ss_model(x_step, u_step)
            k2 = self.ss_model(x_step + dt2 * k1, u_step)
            k3 = self.ss_model(x_step + dt2 * k2, u_step)
            k4 = self.ss_model(x_step + self.ts * k3, u_step)
            dx = self.ts / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            x_step = x_step + dx

        X_sim = torch.stack(X_sim_list, 0)

        return X_sim