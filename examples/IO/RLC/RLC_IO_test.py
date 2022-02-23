import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from torchid.io.module.io_simulator import NeuralIOSimulator
from torchid.io.module.iomodels import NeuralIOModel
import torchid.metrics as metrics

if __name__ == '__main__':

    n_a = 2  # autoregressive coefficients for y
    n_b = 2  # autoregressive coefficients for u
    plot_input = False

    #dataset_type = 'id'
    dataset_type = 'test'
#    model_type = '16step_noise'
#    model_type = '32step_noise'
#    model_type = '64step_noise
    model_type = '1step_nonoise'
#    model_type = '1step_noise'

    # Create fig folder if it does not exist
    if not os.path.exists("fig"):
        os.makedirs("fig")

    # Column names in the dataset
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    # Load dataset
    dataset_filename = f"RLC_data_{dataset_type}.csv"
    df_X = pd.read_csv(os.path.join("data", dataset_filename))
    time_data = np.array(df_X[COL_T], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    y_var_idx = 0  # 0: voltage 1: current
    y = np.copy(x[:, [y_var_idx]])
    N = np.shape(y)[0]
    Ts = time_data[1] - time_data[0]

    # Add measurement noise
    std_noise_V = 10.0
    std_noise_I = 1.0
    std_noise = np.array([std_noise_V, std_noise_I])
    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = x_noise[:, [y_var_idx]]

    # Build validation data
    t_val_start = 0
    t_val_end = time_data[-1]
    idx_val_start = int(t_val_start//Ts)#x.shape[0]
    idx_val_end = int(t_val_end//Ts)#x.shape[0]
    n_val = idx_val_end - idx_val_start
    u_val = np.copy(u[idx_val_start:idx_val_end])
    y_val = np.copy(y[idx_val_start:idx_val_end])
    y_meas_val = np.copy(y_noise[idx_val_start:idx_val_end])
    time_val = time_data[idx_val_start:idx_val_end]

    # Setup neural model structure and load fitted model parameters
    io_model = NeuralIOModel(n_a=n_a, n_b=n_b, n_feat=64)
    io_solution = NeuralIOSimulator(io_model)
    model_filename = f"model_IO_{model_type}.pt"
    io_solution.io_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # Evaluate the model in open-loop simulation against validation data
    y_seq = np.zeros(n_a, dtype=np.float32)
    u_seq = np.zeros(n_b, dtype=np.float32 )
    y_meas_val_torch = torch.tensor(y_meas_val)
    with torch.no_grad():
        y_seq_torch = torch.tensor(y_seq)
        u_seq_torch = torch.tensor(u_seq)

        u_torch = torch.tensor(u_val)
        y_val_sim_torch = io_solution.f_sim(y_seq_torch, u_seq_torch, u_torch)

        err_val = y_val_sim_torch - y_meas_val_torch
        loss_val = torch.mean(err_val**2)

    # Plot results
    if dataset_type == 'id':
        t_plot_start = 0.2e-3
    else:
        t_plot_start = 1.0e-3
    t_plot_end = t_plot_start + 0.3e-3

    idx_plot_start = int(t_plot_start//Ts)#x.shape[0]
    idx_plot_end = int(t_plot_end//Ts)#x.shape[0]

    y_val_sim = np.array(y_val_sim_torch)
    time_val_us = time_val*1e6

    if plot_input:
        fig, ax = plt.subplots(2, 1, sharex=True)
    else:
        fig, ax = plt.subplots(1, 1, sharex=True)
        ax = [ax]

    ax[0].plot(time_val_us[idx_plot_start:idx_plot_end], y_val[idx_plot_start:idx_plot_end], 'k', label='True')
    ax[0].plot(time_val_us[idx_plot_start:idx_plot_end], y_val_sim[idx_plot_start:idx_plot_end], 'r--',  label='Model simulation')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_xlabel("Time ($\mu$s)")
    ax[0].set_ylabel("Capacitor voltage $v_C$ (V)")
    ax[0].set_ylim([-400, 400])

    if plot_input:
        ax[1].plot(time_val_us[idx_plot_start:idx_plot_end], u_val[idx_plot_start:idx_plot_end], 'k', label='Input')
        #ax[1].legend()
        ax[1].grid(True)
        ax[1].set_xlabel("Time ($\mu$s)")
        ax[1].set_ylabel("Input voltage $v_{in}$ (V)")

    fig_name = f"RLC_IO_{dataset_type}_{model_type}.pdf"
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    # R-squared metrics
    R_sq = metrics.r_squared(y_val, y_val_sim)
    print(f"R-squared metrics: {R_sq}")
