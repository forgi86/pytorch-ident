import torch
from torch.utils.data import Dataset, TensorDataset


class SubsequenceDataset(Dataset):
    def __init__(self, input, output, subseq_len):
        self.input = input
        self.output = output
        self.subseq_len = subseq_len
        self.length = self.input.shape[0]

    def __len__(self):
        return self.length - self.subseq_len

    def __getitem__(self, idx):
        return self.input[idx:idx+self.subseq_len], self.output[idx:idx+self.subseq_len]
