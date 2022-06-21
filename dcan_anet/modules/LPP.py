import torch.nn as nn


class LocalPointPred(nn.Module):
    def __init__(self, dim):
        super(LocalPointPred, self).__init__()
        self.dim = dim
        self.start_point_pred = nn.Sequential(
            nn.Conv1d(self.dim, self.dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv1d(self.dim, 1, kernel_size=1), nn.Sigmoid())
        self.end_point_pred = nn.Sequential(
            nn.Conv1d(self.dim, self.dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv1d(self.dim, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        start = self.start_point_pred(x)
        end = self.end_point_pred(x)
        return start, end
