import torch
import torch.nn as nn
import torch.nn.functional as F

Norm = nn.BatchNorm1d


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


def conv1d_norm(in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        dilation=dilation, bias=False))
    result.add_module("norm", Norm(out_channels))
    return result


class MPTC(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, groups=1, activation=nn.ReLU(),
                 use_se=False,
                 norm=Norm):
        super(MPTC, self).__init__()
        self.conv1 = conv1d_norm(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                 groups=groups)
        self.conv3 = conv1d_norm(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation,
                                 dilation=dilation,
                                 groups=groups)

        self.activation = activation
        self.self_add = in_channels == out_channels
        # self.norm = LayerNorm1d(in_channels)
        # self.norm = nn.BatchNorm1d(in_channels)
        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        res = 0
        if self.self_add:
            res = self.norm(x)
        # return self.activation(self.conv3(x)) + self.activation(self.conv1(x)) + res
        return self.activation(self.conv3(x) + self.conv1(x) + res)


class MTCA(nn.Module):
    def __init__(self, cfg):
        super(MTCA, self).__init__()
        self.dim = cfg.temporal_dim
        self.layer_num = cfg.mtca_layer_num
        self.conv = nn.Sequential(
            nn.Conv1d(self.dim, self.dim, kernel_size=1, stride=1), nn.ReLU())

        self.layers = nn.Sequential(
            *[nn.Sequential(MPTC(self.dim, self.dim, 2 ** (i + 1), groups=1), MPTC(self.dim, self.dim, 3, groups=1)) for
              i in
              range(self.layer_num)])

    def forward(self, x):
        x = self.conv(x)
        x = self.layers(x)
        return x
