import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import scipy.linalg
from torchid.io.module.io_simulator import NeuralIOSimulator
from torchid.io.module.iomodels import NeuralIOModel

if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    num_iter = 15000  # number of iterations
    seq_len = 32  # subsequence length m
    alpha = 0.5  # fit/consistency trade-off constant
    lr = 1e-3  # learning rate
    t_fit = 2e-3  # fit on 2 ms of data
    test_freq = 100  # print message every test_freq iterations
    n_a = 2  # autoregressive coefficients for y
    n_b = 2  # autoregressive coefficients for u
    add_noise = True

    # Column names in the dataset
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']
    df_X = pd.read_csv(os.path.join("data", "RLC_data_id.csv"))

    # Load dataset
    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    # Add measurement noise
    std_noise_V = add_noise * 10.0
    std_noise_I = add_noise * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])
    x_noise = np.copy(x) + np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = x_noise[:, [0]]

    # Get fit data
    N = np.shape(y)[0]
    Ts = t[1] - t[0]
    n_fit = int(t_fit//Ts) #
    n_max = np.max((n_a, n_b)) # delay
    batch_size = (n_fit - n_a) // seq_len

    # Build fit data
    u_fit = u[0:n_fit]
    y_fit = y[0:n_fit]
    y_meas_fit = y_noise[0:n_fit]

    y_hidden_fit_init = np.vstack((np.zeros(n_a).reshape(-1, 1), np.copy(y_meas_fit))).astype(np.float32)
    y_hidden_fit_init_true = np.vstack((np.zeros(n_a).reshape(-1, 1), np.copy(y_fit))).astype(np.float32)  # not used, just for reference
    v_fit = np.copy(u_fit)
    v_fit = np.vstack((np.zeros(n_b).reshape(-1, 1), v_fit)).astype(np.float32)
    phi_fit_u = scipy.linalg.toeplitz(v_fit, v_fit[0:n_a])[n_max - 1:-1, :]  # used for the initial conditions on u

    # To pytorch tensors
    y_hidden_fit_torch = torch.tensor(y_hidden_fit_init, requires_grad=True)  # hidden state. It is an optimization variable!
    y_meas_fit_torch = torch.tensor(y_meas_fit)
    u_fit_torch = torch.tensor(u_fit)

    # Setup neural model structure
    io_model = NeuralIOModel(n_a=n_a, n_b=n_b, n_feat=64)
    io_solution = NeuralIOSimulator(io_model)

    # Setup optimizer
    params_net = list(io_solution.io_model.parameters())
    params_hidden = [y_hidden_fit_torch]
    optimizer = optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_hidden, 'lr': 10*lr},
    ], lr=lr)
    
#    params = list(io_solution.io_model.parameters()) + [y_hidden_fit_torch]
#    optimizer = optim.Adam(params, lr=lr)

    # Batch extraction function
    def get_batch(batch_size, seq_len):

        # Select batch indexes
        num_train_samples = y_meas_fit_torch.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False) # batch start indices
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len) # batch all indices
        batch_idx_initial_cond_y = batch_start[:, np.newaxis] - 1 - np.arange(n_a)

        # Extract batch data
        batch_y_hidden_initial_cond = y_hidden_fit_torch[[batch_idx_initial_cond_y + n_a]].squeeze()  # hidden y initial condition for all batch instances
        batch_u_initial_cond = torch.tensor(phi_fit_u[batch_start])  # u initial condition for all batch instances
        batch_y_meas = torch.tensor(y_meas_fit[batch_idx])
        batch_u = torch.tensor(u_fit[batch_idx])
        batch_y_hidden = y_hidden_fit_torch[[batch_idx + n_a]]

        return batch_u, batch_y_meas, batch_y_hidden, batch_y_hidden_initial_cond, batch_u_initial_cond, batch_start

    def get_sequential_batch(seq_len):

        # Select batch indexes
        num_train_samples = y_meas_fit_torch.shape[0]
        batch_size = num_train_samples//seq_len
        batch_start = np.arange(0, batch_size, dtype=np.int64) * seq_len
        batch_idx = batch_start[:,np.newaxis] + np.arange(seq_len) # batch all indices
        batch_idx_initial_cond_y =  batch_start[:, np.newaxis] - 1 - np.arange(n_a)

        # Extract batch data
        batch_y_hidden_initial_cond = y_hidden_fit_torch[[batch_idx_initial_cond_y + n_a]].squeeze()  # hidden y initial condition for all batch instances
        batch_u_initial_cond = torch.tensor(phi_fit_u[batch_start])  # u initial condition for all batch instances
        batch_y_meas = torch.tensor(y_meas_fit[batch_idx]) # batch measured output
        batch_u = torch.tensor(u_fit[batch_idx])           # batch input
        batch_y_hidden = y_hidden_fit_torch[[batch_idx + n_a]]    # batch hidden output

        return batch_u, batch_y_meas, batch_y_hidden, batch_y_hidden_initial_cond, batch_u_initial_cond, batch_start

    # Scale loss with respect to the initial one
    with torch.no_grad():
        batch_u, batch_y_meas, batch_y_hidden, batch_y_hidden_initial_cond, batch_u_initial_cond, batch_start = get_batch(batch_size, seq_len)
        batch_y_sim = io_solution.f_sim_multistep(batch_u, batch_y_hidden_initial_cond, batch_u_initial_cond)
        err_fit = batch_y_meas - batch_y_sim
        loss_fit = torch.mean(err_fit ** 2)
        loss_scale = np.float32(loss_fit)

    LOSS = []
    start_time = time.time()
    # Training loop
    for itr in range(0, num_iter):
        optimizer.zero_grad()

        # Simulate
        batch_u, batch_y_meas, batch_y_hidden, batch_y_hidden_initial_cond, batch_u_initial_cond, batch_start = get_batch(batch_size, seq_len)
        batch_y_sim = io_solution.f_sim_multistep(batch_u, batch_y_hidden_initial_cond, batch_u_initial_cond)

        # Compute fit loss
        err_fit = batch_y_sim - batch_y_meas
        loss_fit = torch.mean(err_fit ** 2)
        loss_fit_sc = loss_fit / loss_scale

        # Compute consistency loss
        err_consistency = batch_y_sim - batch_y_hidden
        loss_consistency = torch.mean(err_consistency ** 2)
        loss_consistency_sc = loss_consistency / loss_scale

        # Compute trade-off loss
        loss_sc = alpha*loss_fit_sc + (1.0-alpha)*loss_consistency_sc

        # Statistics
        LOSS.append(loss_sc.item())
        if itr % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {itr} | Tradeoff Loss {loss_sc:.4f}   Consistency Loss {loss_consistency:.4f}   Fit Loss {loss_fit:.4f}')

        # Optimization step
        loss_fit_sc.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")  # 325 for 15000 iter

    if not os.path.exists("models"):
        os.makedirs("models")
    if add_noise:
        model_filename = f"model_IO_{seq_len}step_noise.pt"
    else:
        model_filename = f"model_IO_{seq_len}step_nonoise.pt"

    torch.save(io_solution.io_model.state_dict(), os.path.join("models", model_filename))

    # Build validation data
    n_val = N
    u_val = u[0:n_val]
    y_val = y[0:n_val]
    y_meas_val = y_noise[0:n_val]

    # Neglect initial values
    y_val = y_val[n_max:, :]
    y_meas_val = y_meas_val[n_max:, :]
    u_val = u_val[n_max:, :]

    y_meas_val_torch = torch.tensor(y_meas_val)

    with torch.no_grad():
        y_seq = np.array(np.flip(y_val[0:n_a].ravel()))
        y_seq_torch = torch.tensor(y_seq)

        u_seq = np.array(np.flip(u_val[0:n_b].ravel()))
        u_seq_torch = torch.tensor(u_seq)

        u_torch = torch.tensor(u_val[n_max:, :])
        y_val_sim_torch = io_solution.f_sim(y_seq_torch, u_seq_torch, u_torch)

        err_val = y_val_sim_torch - y_meas_val_torch[n_max:, :]
        loss_val = torch.mean(err_val**2)

    if not os.path.exists("fig"):
        os.makedirs("fig")

    # Plot the model simulation vs true noise-free values
    if not os.path.exists("fig"):
        os.makedirs("fig")

    y_val_sim = np.array(y_val_sim_torch)
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(y_val, 'b', label='True')
    ax[0].plot(y_val_sim, 'r',  label='Sim')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(u_val, label='Input')
    ax[1].legend()
    ax[1].grid(True)

    if add_noise:
        fig_name = f"RLC_IO_loss_{seq_len}step_noise.pdf"
    else:
        fig_name = f"RLC_IO_loss_{seq_len}step_nonoise.pdf"

    # Plot the training loss
    fig, ax = plt.subplots(1,1, figsize=(7.5, 6))
    ax.plot(np.array(LOSS)/LOSS[0])
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    y_hidden_fit_optimized = y_hidden_fit_torch.detach().numpy()

    fig, ax = plt.subplots(1, 1, sharex=True)
    ax = [ax]
    ax[0].plot(y_hidden_fit_init_true, 'k', label='True')
    ax[0].plot(y_hidden_fit_init, 'b', label='Measured')
    ax[0].plot(y_hidden_fit_optimized, 'r', label='Hidden')
    ax[0].legend()
    ax[0].grid(True)

