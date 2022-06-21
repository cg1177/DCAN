import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self, in_dim, dim):
        super(BaseNet, self).__init__()
        self.in_dim = in_dim
        self.dim = dim

        self.net = nn.Sequential(nn.Conv1d(self.in_dim, self.dim, kernel_size=3, padding=1, groups=4),
                                 nn.ReLU(),
                                 nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1, groups=4),
                                 nn.ReLU(),
                                 nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1, groups=4),
                                 nn.ReLU())

    def forward(self, x):
        x = self.net(x)
        return x
