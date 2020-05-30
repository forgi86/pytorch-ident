import matplotlib
matplotlib.use("TkAgg")
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join("..", ".."))
from torchid.ssmodels_ct import NeuralStateSpaceModel
from torchid.ss_simulator_ct import ForwardEulerSimulator

# Truncated simulation error minimization method
if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    num_iter = 10000  # gradient-based optimization steps
    seq_len = 64  # subsequence length m
    batch_size = 64 # batch size q
    t_fit = 2e-3  # fitting on t_fit ms of data
    alpha = 1.0  # regularization weight
    lr = 1e-3  # learning rate
    test_freq = 100  # print message every test_freq iterations
    add_noise = True

    # Column names in the dataset
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "RLC_data_id.csv"))
    t = np.array(df_X[COL_T], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    # Add measurement noise
    std_noise_V = add_noise * 10.0
    std_noise_I = add_noise * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])
    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)

    # Compute SNR
    P_x = np.mean(x ** 2, axis=0)
    P_n = std_noise**2
    SNR = P_x/(P_n+1e-10)
    SNR_db = 10*np.log10(SNR)

    # Get fit data #
    ts = t[1] - t[0]
    n_fit = int(t_fit // ts)  # x.shape[0]
    u_fit = u[0:n_fit]
    x_fit = x_noise[0:n_fit]
    x_fit_nonoise = x[0:n_fit] # not used, just for reference
    time_fit = t[0:n_fit]

    # Fit data to pytorch tensors #
    u_torch_fit = torch.from_numpy(u_fit)
    x_meas_torch_fit = torch.from_numpy(x_fit)
    time_torch_fit = torch.from_numpy(time_fit)
    x_hidden_fit = torch.tensor(x_fit, requires_grad=True)  # hidden state is an optimization variable

    # Setup neural model structure
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)
    nn_solution = ForwardEulerSimulator(ss_model) #ForwardEulerSimulator(ss_model) #ExplicitRKSimulator(ss_model)

    # Setup optimizer
    params_net = list(nn_solution.ss_model.parameters())
    params_hidden = [x_hidden_fit]
    optimizer = optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_hidden, 'lr': 10*lr},
    ], lr=lr)

    # Batch extraction funtion
    def get_batch(batch_size, seq_len):

        # Select batch indexes
        num_train_samples = x_fit.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False) # batch start indices
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len) # batch samples indices
        batch_idx = batch_idx.T  # transpose indexes to obtain batches with structure (m, q, n_x)

        # Extract batch data
        batch_t = torch.tensor(time_fit[batch_idx])
        batch_x0_hidden = x_hidden_fit[batch_start, :]
        batch_x_hidden = x_hidden_fit[[batch_idx]]
        batch_u = torch.tensor(u_fit[batch_idx])
        batch_x = torch.tensor(x_fit[batch_idx])

        return batch_t, batch_x0_hidden, batch_u, batch_x, batch_x_hidden


    # Scale loss with respect to the initial one
    with torch.no_grad():
        batch_t, batch_x0_hidden, batch_u, batch_x, batch_x_hidden = get_batch(batch_size, seq_len)
        batch_x_sim = nn_solution(batch_x0_hidden, batch_u)
        traced_nn_solution = torch.jit.trace(nn_solution, (batch_x0_hidden, batch_u))
        err_init = batch_x_sim - batch_x
        scale_error = torch.sqrt(torch.mean(err_init**2, dim=(0, 1)))

    LOSS = []
    LOSS_CONSISTENCY = []
    LOSS_FIT = []
    start_time = time.time()
    # Training loop

    scripted_nn_solution = torch.jit.script(nn_solution)
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        batch_t, batch_x0_hidden, batch_u, batch_x, batch_x_hidden = get_batch(batch_size, seq_len)
        #batch_x_sim = traced_nn_solution(batch_x0_hidden, batch_u) # 52 seconds RK | 13 FE
        #batch_x_sim = nn_solution(batch_x0_hidden, batch_u) # 70 seconds RK | 13 FE
        batch_x_sim = scripted_nn_solution(batch_x0_hidden, batch_u) # 71 seconds RK | 13 FE

        # Compute fit loss
        err_fit = batch_x_sim - batch_x
        err_fit_scaled = err_fit/scale_error
        loss_fit = torch.mean(err_fit_scaled**2)

        # Compute consistency loss
        err_consistency = batch_x_sim - batch_x_hidden
        err_consistency_scaled = err_consistency/scale_error
        loss_consistency = torch.mean(err_consistency_scaled**2)

        # Compute trade-off loss
        loss = loss_fit + alpha*loss_consistency

        # Statistics
        LOSS.append(loss.item())
        LOSS_CONSISTENCY.append(loss_consistency.item())
        LOSS_FIT.append(loss_fit.item())
        if itr % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {itr} | Tradeoff Loss {loss:.4f}   Consistency Loss {loss_consistency:.4f}   Fit Loss {loss_fit:.4f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}") # 182 seconds

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    if add_noise:
        model_filename = f"model_SS_{seq_len}step_noise.pkl"
        hidden_filename = f"hidden_SS_{seq_len}step_noise.pkl"
    else:
        model_filename = f"model_SS_{seq_len}step_nonoise.pkl"
        hidden_filename = f"hidden_SS_{seq_len}step_nonoise.pkl"

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_filename))
    torch.save(x_hidden_fit, os.path.join("models", hidden_filename))

    t_val = 5e-3
    n_val = int(t_val // ts)  # x.shape[0]

    input_data_val = u[0:n_val]
    state_data_val = x[0:n_val]

    x0_val = np.zeros(2, dtype=np.float32)
    x0_torch_val = torch.from_numpy(x0_val)
    u_torch_val = torch.tensor(input_data_val)
    x_true_torch_val = torch.from_numpy(state_data_val)

    with torch.no_grad():
        x_sim_torch_val = nn_solution(x0_torch_val[None, :], u_torch_val[:, None, :])
        x_sim_torch_val = x_sim_torch_val.squeeze(1)

    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(np.array(x_true_torch_val[:, 0]), label='True')
    ax[0].plot(np.array(x_sim_torch_val[:, 0]), label='Fit')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(np.array(x_true_torch_val[:, 1]), label='True')
    ax[1].plot(np.array(x_sim_torch_val[:, 1]), label='Fit')
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(np.array(u_torch_val), label='Input')
    ax[2].grid(True)

    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS, 'k', label='ALL')
    ax.plot(LOSS_CONSISTENCY, 'r', label='CONSISTENCY')
    ax.plot(LOSS_FIT, 'b', label='FIT')
    ax.grid(True)
    ax.legend()
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    if add_noise:
        fig_name = f"RLC_SS_loss_{seq_len}step_noise.pdf"
    else:
        fig_name = f"RLC_SS_loss_{seq_len}step_nonoise.pdf"

    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    x_hidden_fit_np = x_hidden_fit.detach().numpy()
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x_fit_nonoise[:, 0], 'k', label='True')
    ax[0].plot(x_fit[:, 0], 'b', label='Measured')
    ax[0].plot(x_hidden_fit_np[:, 0], 'r', label='Hidden')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(x_fit_nonoise[:, 1], 'k', label='True')
    ax[1].plot(x_fit[:, 1], 'b', label='Measured')
    ax[1].plot(x_hidden_fit_np[:, 1], 'r', label='Hidden')
    ax[1].legend()
    ax[1].grid(True)

