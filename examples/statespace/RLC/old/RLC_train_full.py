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
sys.path.append(os.path.join("../..", ".."))
from torchid.ssmodels_ct import NeuralStateSpaceModel
from torchid.ss_simulator_ct import ForwardEulerSimulator

# Full simulation error minimization method
if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    num_iter = 10000  # gradient-based optimization steps
    t_fit = 2e-3  # fitting on t_fit ms of data
    lr = 1e-3  # learning rate
    test_freq = 10  # print message every test_freq iterations
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
    Ts = t[1] - t[0]
    n_fit = int(t_fit // Ts)  # x.shape[0]
    u_fit = u[0:n_fit]
    x_fit = x_noise[0:n_fit]
    x_fit_nonoise = x[0:n_fit] # not used, just for reference
    time_fit = t[0:n_fit]

    # Fit data to pytorch tensors #
    u_torch_fit = torch.from_numpy(u_fit)
    x_meas_torch_fit = torch.from_numpy(x_fit)
    time_torch_fit = torch.from_numpy(time_fit)

    # Setup neural model structure
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)
    nn_solution = ForwardEulerSimulator(ss_model)

    # Scale loss with respect to the initial one
    with torch.no_grad():
        x0_torch = torch.tensor([0.0, 0.0])
        x_sim_torch_fit = nn_solution(x0_torch[None, :], u_torch_fit[:, None, :])
        x_sim_torch_fit = x_sim_torch_fit.squeeze(1)
        err_init = x_meas_torch_fit - x_sim_torch_fit
        scale_error = torch.sqrt(torch.mean((err_init)**2, dim=(0)))

    scripted_nn_solution = torch.jit.script(nn_solution)

    # Setup optimizer
    params_net = list(scripted_nn_solution.parameters())
    optimizer = optim.Adam([
        {'params': params_net,    'lr': lr},
    ], lr=lr)

    LOSS = []
    start_time = time.time()
    # Training loop
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        x0_torch = torch.tensor([0.0, 0.0])
        x_sim_torch_fit = nn_solution(x0_torch[None, :], u_torch_fit[:, None, :])
        x_sim_torch_fit = x_sim_torch_fit.squeeze(1)

        # Compute fit loss
        err_fit = x_sim_torch_fit - x_meas_torch_fit
        err_fit_scaled = err_fit/scale_error
        loss = torch.mean(err_fit_scaled**2)

        # Statistics
        LOSS.append(loss.item())
        if itr % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {itr} | Loss {loss:.4f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}") # 8043.92 seconds

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    if add_noise:
        model_filename = f"model_SS_fullsim_noise.pkl"
        hidden_filename = f"hidden_SS_fullsim_noise.pkl"
    else:
        model_filename = f"model_SS_fullsim_nonoise.pkl"
        hidden_filename = f"hidden_SS_fullsim_nonoise.pkl"

    torch.save(nn_solution.f_xu.state_dict(), os.path.join("models", model_filename))


    # In[Simulate]

    t_val = 5e-3
    n_val = int(t_val // Ts)  # x.shape[0]

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
    ax.grid(True)
    ax.legend()
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    if add_noise:
        fig_name = f"RLC_SS_loss_fullsim_noise.pdf"
    else:
        fig_name = f"RLC_SS_loss_fullsim_nonoise.pdf"

    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')
