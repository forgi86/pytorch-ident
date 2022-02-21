import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from torchid.ss.dt.models import NeuralStateUpdate, ChannelsOutput
from torchid.ss.dt.simulator import StateSpaceSimulator
from torchid import metrics
from loader import rlc_loader

if __name__ == '__main__':

    model_filename = "ss_model_ae.pt"
    model_data = torch.load(os.path.join("models", model_filename))
    n_x = 2

    # Column names in the dataset
    t, u, y, x = rlc_loader("test", "nl", noise_std=0.0)
    ts = t[1, 0] - t[0, 0]

    # Setup neural model structure and load fitted model parameters
    f_xu = NeuralStateUpdate(n_x=2, n_u=1, n_feat=50)
    g_x = ChannelsOutput(channels=[0])  # output is channel 0
    model = StateSpaceSimulator(f_xu, g_x)
    model.load_state_dict(model_data["model"])

    # Evaluate the model in open-loop simulation against validation data
    x_0 = torch.zeros((1, n_x), dtype=torch.float32)
    with torch.no_grad():
        y_sim = model(x_0, torch.tensor(u)[:, None, :]).squeeze(1)
    y_sim = y_sim.detach().numpy()

    # Plot results
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 5.5))

    ax[0].plot(t, y, 'k',  label='$v_C$')
    ax[0].plot(t, y_sim, 'b',  label='$\hat v_C$')
    ax[0].plot(t, y-y_sim, 'r',  label='e')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_xlabel("Time (mu_s)")
    ax[0].set_ylabel("Voltage (V)")

    ax[1].plot(t, u, 'k',  label='$v_{in}$')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    ax[1].set_xlabel("Time (mu_s)")
    ax[1].set_ylabel("Voltage (V)")

    plt.show()

    # R-squared metrics
    R_sq = metrics.r_squared(y, y_sim)
    print(f"R-squared metrics: {R_sq}")
