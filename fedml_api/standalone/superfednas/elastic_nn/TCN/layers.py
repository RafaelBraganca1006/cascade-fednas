from torch.nn.parameter import Parameter, UninitializedParameter
from torch import _weight_norm, norm_except_dim
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Conv1dWtNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        padding=0,
        dim=0,
        weight_norm_bool=True,
    ):
        super(Conv1dWtNorm, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.dim = dim
        self.weight_norm_bool = weight_norm_bool
        if weight_norm_bool:
            self.conv = weight_norm(self.conv)

    def get_active_filter(self):
        if self.weight_norm_bool:
            return _weight_norm(self.conv.weight_v, self.conv.weight_g, self.dim)
        return self.conv.weight

    def init_weights(self):
        pass

    def forward(self, x, out_channel=None):
        filters = self.get_active_filter().contiguous()

        y = F.conv1d(
            x,
            filters,
            self.conv.bias,
            self.stride,
            self.padding,
            self.dilation,
            1,
        )
        return y


class MyTemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        middle_channel,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        super(MyTemporalBlock, self).__init__()

        self.conv1 = Conv1dWtNorm(
            n_inputs,
            middle_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # copy three important things conv_g, conv_v
        self.conv2 = Conv1dWtNorm(
            middle_channel,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.init_weights()
        self.conv2.init_weights()
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)
