import torch
import torch.nn as nn


class LSTMStateEstimator(nn.Module):
    """ Black-box estimator from the sequences of (u, y) to x[N-1].
    The estimation is performed by processing (u, y) forward in time.
    """

    def __init__(self, n_u, n_y, n_x, hidden_size=16, batch_first=False, flipped=False):
        super(LSTMStateEstimator, self).__init__()
        self.n_u = n_u
        self.n_y = n_y
        self.n_x = n_x
        self.batch_first = batch_first
        self.flipped = flipped

        self.lstm = nn.LSTM(input_size=n_y + n_u, hidden_size=hidden_size, batch_first=batch_first)
        self.lstm_output = nn.Linear(hidden_size, n_x)
        self.dim_time = 1 if self.batch_first else 0

    def forward(self, u, y):
        uy = torch.cat((u, y), -1)
        if self.flipped:
            uy = uy.flip(self.dim_time)
        _, (hN, cN) = self.lstm(uy)
        xN = self.lstm_output(hN).squeeze(0)
        return xN


class FeedForwardStateEstimator(nn.Module):
    def __init__(self, n_u, n_y, n_x, seq_len, hidden_size=64, batch_first=False):
        super(FeedForwardStateEstimator, self).__init__()
        self.n_u = n_u
        self.n_y = n_y
        self.n_x = n_x
        self.batch_first = batch_first
        self.seq_len = seq_len

        self.est_net = nn.Sequential(
            nn.Linear((n_u + n_y)*seq_len, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_x),
        )

    def forward(self, u, y):
        uy = torch.cat((u, y), -1)
        if not self.batch_first:
            uy = uy.transpose(0, 1)
        feat = uy.flatten(start_dim=1)

        x_est = self.est_net(feat)
        return x_est
