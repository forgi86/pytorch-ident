import torch
import torch.nn as nn


class FlippedLSTMStateEstimator(nn.Module):
    """ Black-box estimator from the sequences of (u, y) to x[0].
    The estimation is performed by processing (u, y) backward in time.
    """
    def __init__(self, n_u=1, n_y=1, n_x=2, hidden_size=16, batch_first=False):
        super(FlippedLSTMStateEstimator, self).__init__()
        self.n_u = n_u
        self.n_y = n_y
        self.n_x = n_x
        self.batch_first = batch_first

        self.lstm = nn.LSTM(input_size=n_y+n_u, hidden_size=hidden_size, batch_first=batch_first)
        self.lstm_output = nn.Linear(hidden_size, n_x)

    def forward(self, u, y):
        uy = torch.cat((u, y), -1)
        uy_rev = uy.flip(0)
        h_rev, (h0, c0) = self.lstm(uy_rev)
        x0 = self.lstm_output(h0)
        return x0