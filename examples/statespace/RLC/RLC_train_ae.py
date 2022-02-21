"""
Use an LSTM encoder network to estimate the initial state (backward in time), then simulate it forward in time.
Overall, the combination of state_estimator + nn_solution may be seen as an autoencoder.
"""


import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchid.statespace.module.models import NeuralStateSpaceModel
from torchid.statespace.module.simulator import StateSpaceSimulator
from torchid.statespace.module.estimators import FlippedLSTMStateEstimator
from torchid.datasets import SubsequenceDataset
from loader import rlc_loader


# Truncated simulation error minimization method
if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    epochs = 100  # gradient-based optimization steps
    seq_len = 64  # subsequence length m
    batch_size = 32  # batch size q
    lr = 1e-4  # learning rate
    n_fit = 5000

    # Load dataset
    t, u, y, x = rlc_loader("train", "nl", noise_std=0.1, n_data=n_fit)

    # Setup neural model structure
    f_xu = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    model = StateSpaceSimulator(f_xu)
    state_estimator = FlippedLSTMStateEstimator(n_u=1, n_y=1, n_x=2)

    train_dataset = SubsequenceDataset(torch.from_numpy(u), torch.from_numpy(y), subseq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Setup optimizer
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': lr},
        {'params': state_estimator.parameters(), 'lr': lr},
    ], lr=lr)

    LOSS = []
    LOSS_CONSISTENCY = []
    LOSS_FIT = []
    start_time = time.time()
    # Training loop

    for epoch in range(epochs):

        for batch_idx, (batch_u, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()

            # Compute fit loss
            batch_u = batch_u.transpose(0, 1)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1)  # transpose to time_first

            batch_x0 = state_estimator(batch_u, batch_y)[0, :, :]
            batch_x_sim = model(batch_x0, batch_u)
            batch_y_sim = batch_x_sim[..., [0]]

            # Compute consistency loss
            err_ae = batch_y - batch_y_sim
            loss_ae = torch.mean(err_ae**2)

            # Compute trade-off loss
            loss = loss_ae

            # Statistics
            LOSS.append(loss.item())

            # Optimize
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch} | AE Loss {loss:.4f} ')

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    #%%

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")

    model_filename = "ss_model_ae.pt"
    torch.save(f_xu.state_dict(), os.path.join("models", model_filename))

    n_val = n_fit

    #%% Simulate
    with torch.no_grad():
        u_v = torch.tensor(u[:, None, :])
        y_v = torch.tensor(y[:, None, :])
        x0 = state_estimator(u_v, y_v)[0, :, :]
        y_sim = model(x0, u_v)


    #%% Test
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