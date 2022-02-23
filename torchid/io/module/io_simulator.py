import torch
import torch.nn as nn
import numpy as np


class NeuralIOSimulator:
    """ This class implements prediction/simulation methods for the io model structure

     Attributes
     ----------
     io_model: nn.Module
               The neural io model to be fitted
     """

    def __init__(self, io_model):
        self.io_model = io_model

    def f_onestep(self, PHI):
        """ Naive one-step prediction

        Parameters
        ----------
        PHI : Tensor. Size: (N, n_a + n_b)
              Measured io regressor tensor

        Returns
        -------
        Tensor. Size: (N, n_y)
            One-step prediction of the output

        """

        Y_pred = self.io_model(PHI)
        return Y_pred

    def f_sim(self, y_seq, u_seq, U):
        """ Open-loop simulation

        Parameters
        ----------
        y_seq: Tensor. Size: (n_a)
               Initial regressor with past values of y

        u_seq: Tensor. Size: (n_b)
               Initial regressor with past values of u

        U : Tensor. Size: (N, n_u)
            Input sequence tensor

        Returns
        -------
        Tensor. Size: (N, n_y)
            Open-loop simulation of the output

        """
        N = np.shape(U)[0]
        Y_list = []

        for i in range(N):
            phi = torch.cat((y_seq, u_seq))
            yi = self.io_model(phi)
            Y_list += [yi]

            if i < N-1:
                # y shift
                y_seq[1:] = y_seq[0:-1]
                y_seq[0] = yi

                # u shift
                u_seq[1:] = u_seq[0:-1]
                u_seq[0] = U[i]

        Y = torch.stack(Y_list, 0)
        return Y

    def f_sim_multistep(self, batch_u, batch_y_seq, batch_u_seq):
        """ Multi-step simulation over (mini)batches

        Parameters
        ----------
        batch_u: Tensor. Size: (q, m, n_u)
                 Input sequence for each subsequence in the minibatch

        batch_y_seq: Tensor. Size: (q, n_a)
                 Initial regressor with past values of y for each subsequence in the minibatch

        batch_u_seq: Tensor. Size: (q, n_b)
                 Initial regressor with past values of u for each subsequence in the minibatch

        Returns
        -------
        Tensor. Size: (q, m, n_y)
            Simulated output for all subsequences in the minibatch

        """

        batch_size = batch_u.shape[0] # number of training samples in the batch
        seq_len = batch_u.shape[1] # length of the training sequences
        n_a = batch_y_seq.shape[1] # number of autoregressive terms on y
        n_b = batch_u_seq.shape[1] # number of autoregressive terms on u

        Y_sim_list = []
        for i in range(seq_len):
            phi = torch.cat((batch_y_seq, batch_u_seq), -1)
            yi = self.io_model(phi)
            Y_sim_list += [yi]

            # y shift
            batch_y_seq[:, 1:] = batch_y_seq[:, 0:-1]
            batch_y_seq[:, [0]] = yi[:]

            # u shift
            batch_u_seq[:, 1:] = batch_u_seq[:, 0:-1]
            batch_u_seq[:, [0]] = batch_u[:, i]

        Y_sim = torch.stack(Y_sim_list, 1)
        return Y_sim
