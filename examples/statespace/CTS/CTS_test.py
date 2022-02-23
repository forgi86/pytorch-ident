import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from torchid.ss.dt.simulator import StateSpaceSimulator
from torchid.ss.dt.models import CTSNeuralStateSpace
from examples.util import metrics

if __name__ == '__main__':

    plot_input = False

    dataset_type = 'val'
    #model_name = 'model_ss_256step'
    #hidden_name = 'hidden_ss_256step'

    model_name = 'model_ss_full'
    hidden_name = 'hidden_ss_full'

    # Load dataset
    df_data = pd.read_csv(os.path.join("data", "CascadedTanksFiles", "dataBenchmark.csv"))
    if dataset_type == 'id':
        u = np.array(df_data[['uEst']]).astype(np.float32)
        y = np.array(df_data[['yEst']]).astype(np.float32)
    else:
        u = np.array(df_data[['uVal']]).astype(np.float32)
        y = np.array(df_data[['yVal']]).astype(np.float32)

    ts = df_data['Ts'][0].astype(np.float32)
    time_exp = np.arange(y.size).astype(np.float32) * ts

    # Build validation data
    t_val_start = 0
    t_val_end = time_exp[-1]
    idx_val_start = int(t_val_start//ts)
    idx_val_end = int(t_val_end//ts)

    y_meas_val = y[idx_val_start:idx_val_end]
    u_val = u[idx_val_start:idx_val_end]
    time_val = time_exp[idx_val_start:idx_val_end]

    # Setup neural model structure
    ss_model = CTSNeuralStateSpace(n_x=2, n_u=1, n_feat=64)
    nn_solution = StateSpaceSimulator(ss_model)
    nn_solution.state_update.load_state_dict(torch.load(os.path.join("models", model_name + ".pt")))
    x_hidden_fit = torch.load(os.path.join("models", hidden_name + ".pt"))

    # Evaluate the model in open-loop simulation against validation data
    # initial state had to be estimated, according to the dataset description
    x_0 = x_hidden_fit[0, :].detach().numpy()
    with torch.no_grad():
        x_sim_val_torch = nn_solution(torch.tensor(x_0), torch.tensor(u_val))

    # Transform to numpy arrays
    x_sim_val = x_sim_val_torch.detach().numpy()
    y_sim_val = x_sim_val[:, [0]]

    # Plot results
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 5.5))
    idx_plot_start = 0
    idx_plot_end = time_val.size

    ax[0].plot(time_val[idx_plot_start:idx_plot_end], y_meas_val[idx_plot_start:idx_plot_end, 0], 'k', label='$y$')
    ax[0].plot(time_val[idx_plot_start:idx_plot_end], y_sim_val[idx_plot_start:idx_plot_end, 0], 'r--', label='$\hat{y}^{\mathrm{sim}}$')
    ax[0].legend(loc='upper right')
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Voltage (V)")
    ax[0].grid(True)

    ax[1].plot(time_val[idx_plot_start:idx_plot_end], u_val[idx_plot_start:idx_plot_end, 0], 'k', label='$u$')
    ax[1].legend(loc='upper right')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Voltage (V)")
    #ax[1].set_ylim([-5, 5])
    ax[1].grid(True)

    # Plot all
    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig_name = f"CTS_SS_{dataset_type}_{model_name}.pdf"
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    # R-squared metrics
    R_sq = metrics.r_squared(y_sim_val, y_meas_val)
    rmse_sim = metrics.error_rmse(y_sim_val, y_meas_val)

    print(f"R-squared metrics: {R_sq}")
    print(f"RMSE-squared metrics: {rmse_sim}")
