from torch.nn.parameter import Parameter, UninitializedParameter
from torch import _weight_norm, norm_except_dim
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .layers import MyTemporalBlock
from ofa.utils.pytorch_utils import get_net_device


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class DynamicConv1dWtNorm(nn.Module):
    def __init__(
        self,
        max_in_channels,
        max_out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        padding=0,
        dim=0,
        weight_norm_bool=True,
    ):
        super(DynamicConv1dWtNorm, self).__init__()
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.conv = nn.Conv1d(
            self.max_in_channels,
            self.max_out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        # weight.requires_grad = False
        self.dim = dim

        self.weight_norm_bool = weight_norm_bool
        if weight_norm_bool:
            self.conv = weight_norm(self.conv)
        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, out_channel, in_channel):
        if self.weight_norm_bool:
            return _weight_norm(
                self.conv.weight_v[:out_channel, :in_channel, :],
                self.conv.weight_g[:out_channel, :, :],
                self.dim,
            )
        return self.conv.weight[:out_channel, :in_channel, :]

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()

        y = F.conv1d(
            x,
            filters,
            self.conv.bias[:out_channel],
            self.stride,
            self.padding,
            self.dilation,
            1,
        )
        return y


class DynamicTemporalBlock(nn.Module):
    def __init__(
        self,
        maxin_channel,
        maxout_channel,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
        expand_ratio_list=[1.0],
    ):
        super(DynamicTemporalBlock, self).__init__()
        self.active_expand_ratio = max(expand_ratio_list)
        self.expand_ratio_list = expand_ratio_list
        self.active_out_channel = maxout_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.dropout = dropout
        max_middle_channel = self.active_middle_channels
        self.conv1 = DynamicConv1dWtNorm(
            maxin_channel,
            max_middle_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = DynamicConv1dWtNorm(
            max_middle_channel,
            maxout_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (
            DynamicConv1dWtNorm(
                maxin_channel, maxout_channel, 1, weight_norm_bool=False
            )
            if maxin_channel != maxout_channel
            else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.conv.weight.data.normal_(0, 0.01)
        self.conv2.conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.conv.weight.data.normal_(0, 0.01)

    @property
    def active_middle_channels(self):
        return round(self.active_out_channel * self.active_expand_ratio)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        sub_layer = MyTemporalBlock(
            in_channel,
            self.active_out_channel,
            self.active_middle_channels,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.padding,
            self.dropout,
        )
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        middle_channel = self.active_middle_channels
        out_channel = self.active_out_channel

        sub_layer.conv1.conv.weight_g.data.copy_(
            self.conv1.conv.weight_g.data[:middle_channel, :, :]
        )
        sub_layer.conv1.conv.weight_v.data.copy_(
            self.conv1.conv.weight_v.data[:middle_channel, :in_channel, :]
        )
        sub_layer.conv1.conv.bias.data.copy_(
            self.conv1.conv.bias.data[:middle_channel]
        )

        sub_layer.conv2.conv.weight_g.data.copy_(
            self.conv2.conv.weight_g.data[:out_channel, :, :]
        )
        sub_layer.conv2.conv.weight_v.data.copy_(
            self.conv2.conv.weight_v.data[:out_channel, :middle_channel, :]
        )
        sub_layer.conv2.conv.bias.data.copy_(
            self.conv2.conv.bias.data[:out_channel]
        )
        if self.downsample is not None:
            sub_layer.downsample.weight.data.copy_(self.downsample.conv.weight.data[:out_channel, :in_channel, :])
            sub_layer.downsample.bias.data.copy_(self.downsample.conv.bias.data[:out_channel])
        return sub_layer

    def forward(self, x):
        feature_dim = self.active_middle_channels
        self.conv1.active_out_channel = feature_dim
        self.conv2.active_out_channel = self.active_out_channel
        if self.downsample is not None:
            self.downsample.active_out_channel = self.active_out_channel

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
