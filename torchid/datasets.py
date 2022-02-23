import torch
from torch.utils.data import Dataset, TensorDataset


class SubsequenceDataset(Dataset):
    r"""A dataset returning sub-sequences extracted from longer sequences.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.

    Examples:

        >>> u = torch.randn(1000, 2) # 2 inputs
        >>> y = torch.randn(1000, 3) # 3 outputs
        >>> train_dataset = SubsequenceDataset(u, y, subseq_len=100)
    """

    def __init__(self, *tensors, subseq_len):
        self.tensors = tensors
        self.subseq_len = subseq_len
        self.length = self.tensors[0].shape[0]

    def __len__(self):
        return self.length - self.subseq_len

    def __getitem__(self, idx):
        subsequences = [tensor[idx:idx+self.subseq_len] for tensor in self.tensors]
        return subsequences
