from .dynamic_layers import DynamicTemporalBlock
from ofa.utils import val2list
import random
import torch.nn as nn


class MyTemporalConvNet(nn.Module):
    def __init__(self, blocks):
        super(MyTemporalConvNet, self).__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class DynamicTemporalConvNet(nn.Module):
    def __init__(
        self,
        input_channel,
        num_channels,
        kernel_size=2,
        dropout=0.2,
        depth_list=[2],
        expand_ratio_list=[0.25],
    ):
        super(DynamicTemporalConvNet, self).__init__()
        self.runtime_depth = 0
        self.max_depth = max(depth_list)
        self.depth_list = depth_list
        self.expand_ratio_list = expand_ratio_list
        self.blocks = []
        self.input_channel = input_channel
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_channel if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            self.blocks += [
                DynamicTemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    expand_ratio_list=self.expand_ratio_list,
                )
            ]
        self.blocks = nn.ModuleList(self.blocks)

    def set_active_subnet(self, d=None, e=None, w=None):
        # current elastic elasticTCN doesn't support width multipliers
        if isinstance(d, list):
            d = d[0]
        expand_ratios = val2list(e, len(self.blocks))
        for block, expand_ratio in zip(self.blocks, expand_ratios):
            block.active_expand_ratio = expand_ratio
        if d is not None:
            self.runtime_depth = self.max_depth - d


    def set_max_net(self):
        self.set_active_subnet(
            d=max(self.depth_list), e=max(self.expand_ratio_list), w=None
        )

    def sample_active_subnet(self):
        # current elastic elasticTCN doesn't support width multipliers
        expand_setting = []
        for block in self.blocks:
            expand_setting.append(random.choice(block.expand_ratio_list))
        depth = random.choice(self.depth_list)

        arch_config = {"d": depth, "e": expand_setting, "w": None}
        self.set_active_subnet(**arch_config)
        return arch_config

    def get_active_subnet(self):
        blocks = []
        input_channel = self.input_channel
        for block in self.blocks[: len(self.blocks) - self.runtime_depth]:
            blocks.append(block.get_active_subnet(in_channel=input_channel))
            input_channel = block.active_out_channel

        return MyTemporalConvNet(blocks=blocks)

    def forward(self, x):
        out = x
        for block in self.blocks[: len(self.blocks) - self.runtime_depth]:
            out = block(out)
        return out


class MyTCN(nn.Module):
    def __init__(self, encoder, tcn, decoder, drop, tied_weights=False):
        super(MyTCN, self).__init__()
        self.encoder = encoder
        self.tcn = tcn
        self.decoder = decoder
        self.drop = drop
        if tied_weights:
            self.decoder.weight = self.encoder.weight

    def forward(self, input):
        emb = self.drop(self.encoder(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous()


class ElasticTCN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_channels,
        kernel_size=2,
        dropout=0.3,
        emb_dropout=0.1,
        tied_weights=False,
        depth_list=[2],
        expand_ratio_list=[0.25],
    ):
        super(ElasticTCN, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        self.input_size = input_size
        self.output_size = output_size
        self.tcn = DynamicTemporalConvNet(
            input_size,
            num_channels,
            kernel_size,
            dropout=dropout,
            depth_list=depth_list,
            expand_ratio_list=expand_ratio_list,
        )
        self.last_channel = num_channels[-1]

        self.decoder = nn.Linear(num_channels[-1], output_size)
        self.tied_weights = tied_weights
        if tied_weights:
            if num_channels[-1] != input_size:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def set_max_net(self):
        self.tcn.set_max_net()

    def sample_active_subnet(self):
        return self.tcn.sample_active_subnet()

    def set_active_subnet(self, d=None, e=None, w=None):
        self.tcn.set_active_subnet(d=d, e=e, w=w)

    def get_active_subnet(self):
        active_tcn = self.tcn.get_active_subnet()
        encoder = nn.Embedding(self.output_size, self.input_size)
        encoder.weight.data.copy_(self.encoder.weight.data)
        decoder = nn.Linear(self.last_channel, self.output_size)
        decoder.weight.data.copy_(self.decoder.weight.data)
        decoder.bias.data.copy_(self.decoder.bias.data)

        return MyTCN(
            encoder, active_tcn, decoder, nn.Dropout(self.emb_dropout), tied_weights=self.tied_weights
        )

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.drop(self.encoder(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous()
