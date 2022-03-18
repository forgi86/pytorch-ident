import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchid.datasets import SubsequenceDataset
from torchid.ss.dt.models import PolynomialStateUpdate, LinearOutput
from torchid.ss.dt.simulator import StateSpaceSimulator
from torchid.ss.dt.estimators import LSTMStateEstimator
from loader import silverbox_loader
import matplotlib.pyplot as plt


if __name__ == '__main__':

    no_cuda = False  # no GPU, CPU only training
    threads = 6  # max number of CPU threads

    # Parameters
    n_fit = 40000
    seq_sim_len = 256
    seq_est_len = 32  # estimation sequence length
    batch_size = 64
    lr = 1e-3
    epochs = 10
    n_x = 2
    n_u = 1
    n_y = 1
    d_max = 3

    # CPU/GPU resources
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(threads)

    # Load dataset
    t_train, u_train, y_train = silverbox_loader("train", scale=True)

    #%% Prepare dataset
    load_len = seq_sim_len + seq_est_len
    train_data = SubsequenceDataset(u_train, y_train, subseq_len=load_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    f_xu = PolynomialStateUpdate(n_x, n_u, d_max).to(device)
    g_x = LinearOutput(n_x, n_y).to(device)
    f_xu.poly_coeffs = f_xu.poly_coeffs.to(device)  # TODO find a best way to do this automatically
    model = StateSpaceSimulator(f_xu, g_x).to(device)
    estimator = LSTMStateEstimator(n_u=1, n_y=1, n_x=2).to(device)

    # Setup optimizer
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': lr},
        {'params': estimator.parameters(), 'lr': lr},
    ], lr=lr)

    LOSS = []
    LOSS_CONSISTENCY = []
    LOSS_FIT = []

    start_time = time.time()

    # Training loop
    itr = 0
    model.f_xu.freeze_nl()
    for epoch in range(epochs):
        if epoch >= 5:
            model.f_xu.unfreeze_nl()
        for batch_idx, (batch_u, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()

            # Compute fit loss
            batch_u = batch_u.transpose(0, 1).to(device)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1).to(device)  # transpose to time_first

            # Estimate initial state
            batch_u_est = batch_u[:seq_est_len]
            batch_y_est = batch_y[:seq_est_len]
            batch_x0 = estimator(batch_u_est, batch_y_est)

            # Simulate
            batch_u_fit = batch_u[seq_est_len:]
            batch_y_fit = batch_y[seq_est_len:]
            batch_y_sim = model(batch_x0, batch_u_fit)

            # Compute loss
            loss = torch.nn.functional.mse_loss(batch_y_fit, batch_y_sim)

            # Statistics
            LOSS.append(loss.item())

            # Optimize
            loss.backward()
            optimizer.step()

            if itr % 10 == 0:
                print(f'Iteration {itr} | Train Loss {loss:.4f} ')
            itr += 1

        print(f'Epoch {epoch} | Train Loss {loss:.4f} ')

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    #%% Save model
    if not os.path.exists("models"):
        os.makedirs("models")

    model = model.to("cpu")
    model.f_xu.poly_coeffs = f_xu.poly_coeffs.to("cpu")
    estimator = estimator.to("cpu")
    model_filename = "ss_poly.pt"
    torch.save({"n_x": n_x,
                "n_y": n_y,
                "n_u": n_u,
                "d_max": d_max,
                "model": model.state_dict(),
                "estimator": estimator.state_dict()
                },
               os.path.join("models", model_filename))

    #%% Simulate
    t_full, u_full, y_full = silverbox_loader("full", scale=True)
    with torch.no_grad():
        u_v = torch.tensor(u_full[:, None, :])
        y_v = torch.tensor(y_full[:, None, :])
        x0 = estimator(u_v, y_v)
        y_sim = model(x0, u_v).squeeze(1)

    #%% Test
    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS, 'k', label='ALL')
    ax.grid(True)
    ax.legend()
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(y_full[:, 0], 'k', label='meas')
    ax.grid(True)
    ax.plot(y_sim[:, 0], 'b', label='sim')
