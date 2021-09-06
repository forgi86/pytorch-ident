import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import scipy.linalg
from torchid.IO.module.io_simulator import NeuralIOSimulator
from torchid.IO.module.iomodels import NeuralIOModel

if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    t_fit = 2e-3  # fitting on t_fit ms of data
    n_a = 2  # autoregressive coefficients for y
    n_b = 2  # autoregressive coefficients for u
    lr = 1e-4  # learning rate
    num_iter = 40000  # gradient-based optimization steps
    test_freq = 500  # print message every test_freq iterations
    add_noise = False

    # Column names in the dataset
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']
    df_X = pd.read_csv(os.path.join("data", "RLC_data_id.csv"))

    # Load dataset
    t = np.array(df_X[COL_T], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    y_var_idx = 0  # 0: voltage 1: current
    y = np.copy(x[:, [y_var_idx]])

    # Add measurement noise
    std_noise_V = add_noise * 10.0
    std_noise_I = add_noise * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])
    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = x_noise[:, [y_var_idx]]

    # Build fit data
    n_max = np.max((n_a, n_b))  # delay
    N = np.shape(y)[0]
    Ts = t[1] - t[0]
    n_fit = int(t_fit // Ts)  # x.shape[0]
    u_fit = u[0:n_fit]
    y_fit = y[0:n_fit]
    y_meas_fit = y_noise[0:n_fit]
    phi_fit_y = scipy.linalg.toeplitz(y_meas_fit, y_meas_fit[0:n_a])[n_max - 1:-1, :]  # regressor 1
    phi_fit_u = scipy.linalg.toeplitz(u_fit, u_fit[0:n_a])[n_max - 1:-1, :]
    phi_fit = np.hstack((phi_fit_y, phi_fit_u))

    # Neglect initial values
    y_fit = y_fit[n_max:, :]
    y_meas_fit = y_meas_fit[n_max:, :]
    u_fit = u_fit[n_max:, :]

    # Build fit data
    phi_fit_torch = torch.from_numpy(phi_fit)
    y_meas_fit_torch = torch.from_numpy(y_meas_fit)

    # Setup neural model structure
    io_model = NeuralIOModel(n_a=n_a, n_b=n_b, n_feat=64, small_init=True)
    io_solution = NeuralIOSimulator(io_model)

    # Setup optimizer
    optimizer = optim.Adam(io_solution.io_model.parameters(), lr=lr)

    LOSS = []
    start_time = time.time()
    # Training loop
    for itr in range(1, num_iter + 1):
        optimizer.zero_grad()

        # Perform one-step ahead prediction
        y_est_torch = io_solution.f_onestep(phi_fit_torch)

        # Compute fit loss
        err = y_est_torch - y_meas_fit_torch
        loss = torch.mean(err ** 2)

        # Statistics
        LOSS.append(loss.item())
        if itr % test_freq == 0:
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

        # Optimization step
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time  # 73 seconds
    print(f"\nTrain time: {train_time:.2f}")

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    if add_noise:
        model_filename = "model_IO_1step_noise.pt"
    else:
        model_filename = "model_IO_1step_nonoise.pt"

    torch.save(io_solution.io_model.state_dict(), os.path.join("models", model_filename))

    # In[Validate model]
    t_val_start = 0
    t_val_end = t[-1]
    idx_val_start = int(t_val_start // Ts)  # x.shape[0]
    idx_val_end = int(t_val_end // Ts)  # x.shape[0]

    n_val = idx_val_end - idx_val_start
    u_val = np.copy(u[idx_val_start:idx_val_end])
    y_val = np.copy(y[idx_val_start:idx_val_end])
    y_meas_val = np.copy(y_noise[idx_val_start:idx_val_end])

    y_seq = np.array(np.flip(y_val[0:n_a].ravel()))
    u_seq = np.array(np.flip(u_val[0:n_b].ravel()))

    # Neglect initial values
    y_val = y_val[n_max:, :]
    y_meas_val = y_meas_val[n_max:, :]
    u_val = u_val[n_max:, :]

    y_meas_val_torch = torch.tensor(y_meas_val)

    with torch.no_grad():
        y_seq_torch = torch.tensor(y_seq)
        u_seq_torch = torch.tensor(u_seq)

        u_torch = torch.tensor(u_val)
        y_val_sim_torch = io_solution.f_sim(y_seq_torch, u_seq_torch, u_torch)

        err_val = y_val_sim_torch - y_meas_val_torch
        loss_val = torch.mean((err_val) ** 2)

    # Plot
    if not os.path.exists("fig"):
        os.makedirs("fig")
    y_val_sim = np.array(y_val_sim_torch)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(y_val, 'b', label='True')
    ax[0].plot(y_val_sim, 'r', label='Sim')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(u_val, label='Input')
    ax[1].legend()
    ax[1].grid(True)

    if add_noise:
        fig_name = "RLC_IO_loss_1step_noise.pdf"
    else:
        fig_name = "RLC_IO_loss_1step_nonoise.pdf"

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    ax.plot(LOSS)
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')
