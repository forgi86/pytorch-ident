import torch
import torch.nn as nn
import numpy as np
 
        
class NeuralStateSpaceSimulator:
    """ This class implements prediction/simulation methods for the SS model structure

     Attributes
     ----------
     ss_model: nn.Module
               The neural SS model to be fitted
     Ts: float
         model sampling time

     """

    def __init__(self, ss_model, Ts=1.0):
        self.ss_model = ss_model
        self.Ts = Ts

    def f_onestep(self, X, U):
        """ Naive one-step prediction

        Parameters
        ----------
        X : Tensor. Size: (N, n_x)
            State sequence tensor

        U : Tensor. Size: (N, n_u)
            Input sequence tensor

        Returns
        -------
        Tensor. Size: (N, n_x)
            One-step prediction over N steps

        """

        X_pred = torch.empty(X.shape)
        X_pred[0, :] = X[0, :]
        DX = self.ss_model(X[0:-1], U[0:-1])
        X_pred[1:,:] = X[0:-1, :] + DX

        return X_pred

    def f_sim(self, x0, u):
        """ Open-loop simulation

        Parameters
        ----------
        x0 : Tensor. Size: (n_x)
             Initial state

        U : Tensor. Size: (N, n_u)
            Input sequence tensor

        Returns
        -------
        Tensor. Size: (N, n_x)
            Open-loop model simulation over N steps

        """

        N = np.shape(u)[0]
        nx = np.shape(x0)[0]

        X_list = []
        xstep = x0
        for i in range(N):
            X_list += [xstep]
            #X[i,:] = xstep
            ustep = u[i]
            dx = self.ss_model(xstep, ustep)
            xstep = xstep + dx

        X = torch.stack(X_list, 0)

        return X

    def f_sim_multistep(self, x0_batch, U_batch):
        """ Multi-step simulation over (mini)batches

        Parameters
        ----------
        x0_batch: Tensor. Size: (q, n_x)
             Initial state for each subsequence in the minibatch

        U_batch: Tensor. Size: (q, m, n_u)
            Input sequence for each subsequence in the minibatch

        Returns
        -------
        Tensor. Size: (q, m, n_x)
            Simulated state for all subsequences in the minibatch

        """

        batch_size = x0_batch.shape[0]
        n_x = x0_batch.shape[1]
        seq_len = U_batch.shape[1]

        X_sim_list = []
        xstep = x0_batch
        for i in range(seq_len):
            X_sim_list += [xstep] #X_sim[:, i, :] = xstep
            ustep = U_batch[:, i, :]
            dx = self.ss_model(xstep, ustep)
            xstep = xstep + dx

        X_sim = torch.stack(X_sim_list, 1)#.squeeze(2)
        return X_sim

#    def f_residual_fullyobserved(self, X_batch, U_batch):
#        X_increment = X_batch[:, -1, :] - X_batch[:, 0, :]
