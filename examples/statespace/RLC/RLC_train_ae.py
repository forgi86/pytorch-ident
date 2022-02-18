"""
Use an LSTM encoder network to estimate the initial state (backward in time), then simulate it forward in time.
Overall, the combination of state_estimator + nn_solution may be seen as an autoencoder.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
import matplotlib.pyplot as plt


class LSTMFlippedStateEstimator(nn.Module):
    """ Black-box estimator from u, y to x(0)"""
    def __init__(self, n_u=1, n_y=1, n_x=2, batch_first=False):
        super(LSTMFlippedStateEstimator, self).__init__()
        self.n_u = n_u
        self.n_y = n_y
        self.n_x = n_x

        self.lstm = nn.LSTM(input_size=n_y+n_u, hidden_size=16,
                            proj_size=n_x, batch_first=batch_first)

    def forward(self, u, y):
        uy = torch.cat((u, y), -1)
        uy_rev = uy.flip(0)
        x_rev, (x0, c0) = self.lstm(uy_rev)
        return x0


# Truncated simulation error minimization method
if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    num_iter = 20000  # gradient-based optimization steps
    seq_len = 64  # subsequence length m
    batch_size = 32  # batch size q
    t_fit = 2e-3  # fitting on t_fit ms of data
    lr = 1e-4  # learning rate
    test_freq = 100  # print message every test_freq iterations
    add_noise = True

    # Column names in the dataset
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    idx_out = 0  # output=vc

    scale_u = np.array(80., dtype=np.float32)
    scale_x = np.array([90., 3.], dtype=np.float32)

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "RLC_data_id.csv"))
    time_data = np.array(df_X[COL_T], dtype=np.float32)
    t = np.array(df_X[COL_T], dtype=np.float32)

    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    # Add measurement noise
    std_noise_V = add_noise * 10.0
    std_noise_I = add_noise * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])
    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = x_noise[:, [idx_out]]

    x = x/scale_x
    u = u/scale_u
    x_noise = x_noise/scale_x
    y = np.copy(x[:, [idx_out]])


    # Get fit data #
    ts = t[1] - t[0]
    n_fit = int(t_fit // ts)  # x.shape[0]
    u_fit = u[0:n_fit]
    x_fit = x_noise[0:n_fit]
    x_fit_nonoise = x[0:n_fit]  # not used, just for reference
    time_fit = t[0:n_fit]
    y_fit = x_fit[:, [idx_out]]

    # Fit data to pytorch tensors #
    u_torch_fit = torch.from_numpy(u_fit)
    time_torch_fit = torch.from_numpy(time_fit)

    # Setup neural model structure
    state_estimator = LSTMFlippedStateEstimator(n_u=1, n_y=1, n_x=2)
    # Setup neural model structure
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    nn_solution = ForwardEulerSimulator(ss_model)

    # Setup optimizer
    optimizer = optim.Adam([
        {'params': state_estimator.parameters(), 'lr': lr},
        {'params': nn_solution.parameters(), 'lr': lr},
    ], lr=lr)

    # Batch extraction funtion
    def get_batch(batch_size, seq_len):

        # Select batch indexes
        num_train_samples = y_fit.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64),
                                       batch_size, replace=False)  # batch start indices
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len)  # batch samples indices
        batch_idx = batch_idx.T  # transpose indexes to obtain batches with structure (m, q, n_x)

        # Extract batch data
        batch_t = torch.tensor(time_fit[batch_idx])
        batch_u = torch.tensor(u_fit[batch_idx])
        batch_y = torch.tensor(y_fit[batch_idx])

        return batch_t, batch_u, batch_y


    LOSS = []
    LOSS_CONSISTENCY = []
    LOSS_FIT = []
    start_time = time.time()
    # Training loop

    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        batch_t, batch_u, batch_y = get_batch(batch_size, seq_len)

        # Compute fit loss
        batch_x0 = state_estimator(batch_u, batch_y)[0, :, :]
        batch_x_sim = nn_solution(batch_x0, batch_u)
        batch_y_sim = batch_x_sim[..., [0]]

        # Compute consistency loss
        err_ae = batch_y - batch_y_sim
        loss_ae = torch.mean(err_ae**2)

        # Compute trade-off loss
        loss = loss_ae

        # Statistics
        LOSS.append(loss.item())
        if itr % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {itr} | AE Loss {loss:.4f} ')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    #%%

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")

    model_filename = "ss_model_ae.pt"
    torch.save(ss_model.state_dict(), os.path.join("models", model_filename))

    t_val = 5e-3
    n_val = int(t_val // ts)  # x.shape[0]

    #%%
    with torch.no_grad():
        u_v = torch.tensor(u[:, None, :])
        y_v = torch.tensor(y[:, None, :])
        x0 = state_estimator(u_v, y_v)[0, :, :]
        y_sim = nn_solution(x0, u_v)


    #%%
    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS, 'k', label='ALL')
    ax.grid(True)
    ax.legend()
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(y_v[:, 0, 0], 'k', label='meas')
    ax.grid(True)
    ax.plot(y_sim[:, 0, 0], 'b', label='sim')
