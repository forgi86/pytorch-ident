import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torchid.statespace.module.simulator import StateSpaceSimulator
from torchid.statespace.module.models import CTSNeuralStateSpaceModel


if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    num_iter = 10000  # gradient-based optimization steps
    lr = 1e-4  # learning rate
    test_freq = 10  # print message every test_freq iterations

    # Load dataset
    df_data = pd.read_csv(os.path.join("data", "CascadedTanksFiles", "dataBenchmark.csv"))
    u_id = np.array(df_data[['uEst']]).astype(np.float32)
    y_id = np.array(df_data[['yEst']]).astype(np.float32)
    ts = df_data['Ts'][0].astype(np.float32)
    time_exp = np.arange(y_id.size).astype(np.float32)*ts

    x_est = np.zeros((time_exp.shape[0], 2), dtype=np.float32)
    x_est[:, 0] = np.copy(y_id[:, 0])

    # Hidden state variable
    x_hidden_fit = torch.tensor(x_est, dtype=torch.float32, requires_grad=True)  # hidden state is an optimization variable
    y_fit = y_id
    u_fit = u_id
    u_fit_torch = torch.tensor(u_fit)
    y_fit_torch = torch.tensor(y_fit)
    time_fit = time_exp

    # Setup neural model structure
    ss_model = CTSNeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)
    nn_solution = StateSpaceSimulator(ss_model)

    model_name = 'model_SS_256step'
    hidden_name = 'hidden_SS_256step'
    #nn_solution.load_state_dict(torch.load(os.path.join("models", model_name + ".pkl")))
    #x_hidden_fit = torch.load(os.path.join("models", hidden_name + ".pkl"))


    # Setup optimizer
    params_net = list(nn_solution.parameters())
    params_hidden = [x_hidden_fit]
    optimizer = optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_hidden, 'lr': lr},
    ], lr=lr)

    # Scale loss with respect to the initial one
    with torch.no_grad():
        x0_torch = x_hidden_fit[0, :]
        x_est_torch = nn_solution.f_sim(x0_torch, u_fit_torch)
        err_init = x_est_torch[:, [0]] - y_fit_torch
        scale_error = torch.sqrt(torch.mean((err_init)**2, dim=(0)))

    LOSS_TOT = []
    LOSS_FIT = []
    LOSS_CONSISTENCY = []
    start_time = time.time()
    # Training loop

    #scripted_nn_solution = torch.jit.script(nn_solution)
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        x0_torch = x_hidden_fit[0, :]

        # Perform open-loop simulation
        x_sim = nn_solution.f_sim(x0_torch, u_fit_torch)

        # Compute fit loss
        err_fit = x_sim[:, [0]] - y_fit_torch
        err_fit_scaled = err_fit/scale_error[0]
        loss_fit = torch.mean(err_fit_scaled**2)


        # Compute trade-off loss
        loss = loss_fit

        LOSS_TOT.append(loss.item())
        LOSS_FIT.append(loss_fit.item())
        if itr % test_freq == 0:
            print(f'Iter {itr} | Fit Loss {loss_fit:.4f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}") # 182 seconds

    if not os.path.exists("models"):
        os.makedirs("models")

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")

    model_filename = f"model_SS_{'simerr'}.pkl"
    hidden_filename = f"hidden_SS_{'simerr'}.pkl"

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_filename))
    torch.save(x_hidden_fit, os.path.join("models", hidden_filename))

    # Plot figures
    if not os.path.exists("fig"):
        os.makedirs("fig")

    # Loss plot
    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS_TOT, 'k', label='TOT')
    ax.plot(LOSS_CONSISTENCY, 'r', label='CONSISTENCY')
    ax.plot(LOSS_FIT, 'b', label='FIT')
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    fig_name = f"CTS_SS_loss_{'simerr'}_noise.pdf"
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    # Hidden variable plot
    x_hidden_fit_np = x_hidden_fit.detach().numpy()
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(y_id[:, 0], 'b', label='Measured')
    ax[0].plot(x_hidden_fit_np[:, 0], 'r', label='Hidden')
    ax[0].legend()
    ax[0].grid(True)

    #ax[1].plot(x_est[:, 1], 'k', label='Estimated')
    ax[1].plot(x_hidden_fit_np[:, 1], 'r', label='Hidden')
    ax[1].legend()
    ax[1].grid(True)

    # Simulate
    y_val = np.copy(y_fit)
    u_val = np.copy(u_fit)

    #x0_val = np.array(x_est[0, :])
    #x0_val[1] = 0.0
    x0_val = x_hidden_fit[0, :].detach().numpy() # initial state had to be estimated, according to the dataset description
    x0_torch_val = torch.from_numpy(x0_val)
    u_torch_val = torch.tensor(u_val)

    with torch.no_grad():
        x_sim_torch = nn_solution.f_sim(x0_torch_val[None, :], u_torch_val[:, None, :])
        y_sim_torch = x_sim_torch[:, 0]
        x_sim = y_sim_torch.detach().numpy()


    # Simulation plot
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 7.5))
    #ax[0].plot(time_exp, q_ref,  'k',  label='$q_{\mathrm{ref}}$')
    ax[0].plot(time_exp, y_val, 'k', label='$y_{\mathrm{meas}}$')
    ax[0].plot(time_exp, x_sim[:, 0], 'r', label='$\hat y_{\mathrm{sim}}$')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_ylabel("Voltage (V)")

    ax[1].plot(time_exp, u_id, 'k', label='$u_{in}$')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Voltage (V)")
    ax[1].grid(True)
    ax[1].set_xlabel("Time (s)")
