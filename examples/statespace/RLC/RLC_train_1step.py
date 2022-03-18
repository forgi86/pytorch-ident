"""
Train with a 1-step-ahead prediction model.

+ Very simple and efficient
- Requires full state measurement
- It is not very robust to noise.
"""

import os
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torchid.ss.dt.models import NeuralStateUpdate
from torchid.ss.dt.simulator import StateSpaceSimulator
from loader import rlc_loader

if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    t_fit = 2e-3  # fitting on t_fit ms of data
    lr = 1e-4  # learning rate
    num_iter = 40000  # gradient-based optimization steps
    test_freq = 500  # print message every test_freq iterations
    n_feat = 50

    # Load dataset
    t, u, y, x = rlc_loader("train", "nl", noise_std=0.1, dtype=np.float32)
    n_x = x.shape[-1]
    n_u = u.shape[-1]
    n_y = y.shape[-1]

    ts = t[1] - t[0]
    n_fit = int(t_fit // ts)

    # Fit data to pytorch tensors #
    u_train = torch.tensor(u, dtype=torch.float32)
    x_train = torch.tensor(x, dtype=torch.float32)

    # Setup neural model structure
    f_xu = NeuralStateUpdate(n_x=2, n_u=1, n_feat=n_feat)
    model = StateSpaceSimulator(f_xu)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Scale loss with respect to the initial one
    with torch.no_grad():
        delta_x = x_train[1:, :] - x_train[0:-1, :]
        scale_error = torch.sqrt(torch.mean(delta_x ** 2, dim=0))

    LOSS = []
    start_time = time.time()
    # Training loop
    for itr in range(0, num_iter):
        optimizer.zero_grad()

        # Perform one-step ahead prediction
        delta_x_hat = model.f_xu(x_train[0:-1, :], u_train[0:-1, :])
        delta_x = x_train[1:, :] - x_train[0:-1, :]

        err = delta_x - delta_x_hat
        err_scaled = err/scale_error

        # Compute fit loss
        loss = torch.mean(err_scaled**2)

        # Statistics
        LOSS.append(loss.item())
        if itr % test_freq == 0:
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time  # 114 seconds
    print(f"\nTrain time: {train_time:.2f}")

    #%% Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    model_filename = "ss_model_1step.pt"
    torch.save({"n_x": 2,
                "n_y": 1,
                "n_u": 1,
                "model": model.state_dict(),
                "n_feat": n_feat
                },
               os.path.join("models", model_filename))

    #%% Simulate model

    t_test, u_test, y_test, x_test = rlc_loader("test", "nl", noise_std=0.0, dtype=np.float32)

    with torch.no_grad():
        x0 = torch.zeros((1, n_x), dtype=torch.float32)
        y_sim, x_sim = model(x0, torch.tensor(u_test)[:, None, :], return_x=True)

    y_sim = y_sim.squeeze(1).detach().numpy()
    x_sim = x_sim.squeeze(1).detach().numpy()

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x_test[:, 0], 'k+', label='True')
    ax[0].plot(x_sim[:, 0], 'r', label='Sim')
    ax[0].legend()
    ax[1].plot(x_test[:, 1], 'k+', label='True')
    ax[1].plot(x_sim[:, 1], 'r', label='Sim')
    ax[1].legend()
    ax[0].grid(True)
    ax[1].grid(True)

    #%% Plot loss

    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    ax.plot(LOSS)
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")