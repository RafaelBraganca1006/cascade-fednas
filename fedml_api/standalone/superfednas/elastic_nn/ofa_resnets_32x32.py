import random
import typing
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicConvLayer,
    DynamicLinearLayer,
)
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicResNetBottleneckBlock,
)
from ofa.utils.layers import IdentityLayer, ResidualBlock

# from ofa.imagenet_classification.networks import ResNets
from ofa.utils import make_divisible, val2list, MyNetwork
import torch.nn as nn

import torch
from ofa.utils.layers import (
    set_layer_from_config,
    ConvLayer,
    IdentityLayer,
    LinearLayer,
)
from ofa.utils.layers import ResNetBottleneckBlock, ResidualBlock
from ofa.utils import make_divisible, MyNetwork, MyGlobalAvgPool2d


class ResNets32x32(MyNetwork):

    BASE_DEPTH_LIST = [2, 2, 4, 2]
    STAGE_WIDTH_LIST = [256, 512, 1024, 2048]

    def __init__(self, input_stem, blocks, classifier):
        super(ResNets32x32, self).__init__()

        self.input_stem = nn.ModuleList(input_stem)
        self.blocks = nn.ModuleList(blocks)
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=False)
        self.classifier = classifier

    def forward(self, x):
        for i, layer in enumerate(self.input_stem):
            # shape_earlier = x.size()
            x = layer(x)
            # shape_now = x.size()
            # print(f"[STEM-{i}] From {shape_earlier} -> {shape_now} ")
        #         x = self.max_pooling(x)
        for i, block in enumerate(self.blocks):
            # shape_earlier = x.size()
            x = block(x)
            # shape_now = x.size()
            # print(f"[BLOCK-{i}] From {shape_earlier} -> {shape_now} ")

        # shape_earlier = x.size()
        x = self.global_avg_pool(x)
        # shape_now = x.size()
        # print(f"[POOL-{i}] From {shape_earlier} -> {shape_now} ")

        # shape_earlier = x.size()
        x = self.classifier(x)
        # shape_now = x.size()
        # print(f"[LINEAR-{i}] From {shape_earlier} -> {shape_now} ")
        return x

    @property
    def module_str(self):
        _str = ""
        for layer in self.input_stem:
            _str += layer.module_str + "\n"
        _str += "max_pooling(ks=3, stride=2)\n"
        for block in self.blocks:
            _str += block.module_str + "\n"
        _str += self.global_avg_pool.__repr__() + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": ResNets32x32.__name__,
            "bn": self.get_bn_param(),
            "input_stem": [layer.config for layer in self.input_stem],
            "blocks": [block.config for block in self.blocks],
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        classifier = set_layer_from_config(config["classifier"])

        input_stem = []
        for layer_config in config["input_stem"]:
            input_stem.append(set_layer_from_config(layer_config))
        blocks = []
        for block_config in config["blocks"]:
            blocks.append(set_layer_from_config(block_config))

        net = ResNets32x32(input_stem, blocks, classifier)
        if "bn" in config:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-5)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, ResNetBottleneckBlock) and isinstance(
                m.downsample, IdentityLayer
            ):
                m.conv3.bn.weight.data.zero_()

    @property
    def grouped_block_index(self):
        info_list = []
        block_index_list = []
        for i, block in enumerate(self.blocks):
            if (
                not isinstance(block.downsample, IdentityLayer)
                and len(block_index_list) > 0
            ):
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list

    def load_state_dict(self, state_dict, **kwargs):
        super(ResNets32x32, self).load_state_dict(state_dict)


class OFAResNets32x32_50l(ResNets32x32):
    def __init__(
        self,
        n_classes=1000,
        bn_param=(0.1, 1e-5),
        dropout_rate=0,
        depth_list=2,
        expand_ratio_list=0.25,
        width_mult_list=1.0,
    ):

        self.depth_list = val2list(depth_list)
        self.expand_ratio_list = val2list(expand_ratio_list)
        self.width_mult_list = val2list(width_mult_list)
        # sort
        self.depth_list.sort()
        self.expand_ratio_list.sort()
        self.width_mult_list.sort()

        input_channel = [
            make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            for width_mult in self.width_mult_list
        ]
        mid_input_channel = [
            make_divisible(channel // 2, MyNetwork.CHANNEL_DIVISIBLE)
            for channel in input_channel
        ]

        stage_width_list = ResNets32x32.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = [
                make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
                for width_mult in self.width_mult_list
            ]

        n_block_list = [
            base_depth + max(self.depth_list)
            for base_depth in ResNets32x32.BASE_DEPTH_LIST
        ]
        stride_list = [1, 2, 2, 2]

        input_stem = [
            DynamicConvLayer(
                val2list(3), input_channel, 3, stride=1, use_bn=True, act_func="relu",
            )
        ]

        # blocks
        blocks = []
        for d, width, s in zip(n_block_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                bottleneck_block = DynamicResNetBottleneckBlock(
                    input_channel,
                    width,
                    expand_ratio_list=self.expand_ratio_list,
                    kernel_size=3,
                    stride=stride,
                    act_func="relu",
                    downsample_mode="avgpool_conv",
                )
                blocks.append(bottleneck_block)
                input_channel = width
        # classifier
        classifier = DynamicLinearLayer(
            input_channel, n_classes, dropout_rate=dropout_rate
        )

        super(OFAResNets32x32_50l, self).__init__(input_stem, blocks, classifier)

        # set bn param
        self.set_bn_param(*bn_param)

        # runtime_depth
        self.input_stem_skipping = 0
        self.runtime_depth = [0] * len(n_block_list)

    @property
    def ks_list(self):
        return [3]

    @staticmethod
    def name():
        return "OFAResNets"

    def forward(self, x):
        for i, layer in enumerate(self.input_stem):
            if (
                self.input_stem_skipping > 0
                and isinstance(layer, ResidualBlock)
                and isinstance(layer.shortcut, IdentityLayer)
            ):
                pass
            else:
                # shape_earlier = x.size()
                x = layer(x)
                # shape_now = x.size()
                # print(f"[STEM-{i}] From {shape_earlier} -> {shape_now} ")

        #         shape_earlier = x.size()
        #         x = self.max_pooling(x)
        #         shape_now = x.size()
        #         print(f"[MAX_POOL] From {shape_earlier} -> {shape_now} ")
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                # shape_earlier = x.size()
                x = self.blocks[idx](x)
                # shape_now = x.size()
                # print(
                #     f"[STAGE-{stage_id}-BLOCK-{idx}] From {shape_earlier} -> {shape_now} "
                # )
        # shape_earlier = x.size()
        x = self.global_avg_pool(x)
        # shape_now = x.size()
        # print(f"[GLOBAL_AVG_POOL] From {shape_earlier} -> {shape_now} ")
        # shape_earlier = x.size()
        x = self.classifier(x)
        # shape_now = x.size()
        # print(f"[CLASSIFIER] From {shape_earlier} -> {shape_now} ")
        return x

    @property
    def module_str(self):
        _str = ""
        for layer in self.input_stem:
            if (
                self.input_stem_skipping > 0
                and isinstance(layer, ResidualBlock)
                and isinstance(layer.shortcut, IdentityLayer)
            ):
                pass
            else:
                _str += layer.module_str + "\n"
        _str += "max_pooling(ks=3, stride=2)\n"
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + "\n"
        _str += self.global_avg_pool.__repr__() + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": OFAResNets32x32_50l.__name__,
            "bn": self.get_bn_param(),
            "input_stem": [layer.config for layer in self.input_stem],
            "blocks": [block.config for block in self.blocks],
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError("do not support this function")

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            new_key = key
            if new_key in model_dict:
                pass
            elif ".linear." in new_key:
                new_key = new_key.replace(".linear.", ".linear.linear.")
            elif "bn." in new_key:
                new_key = new_key.replace("bn.", "bn.bn.")
            elif "conv.weight" in new_key:
                new_key = new_key.replace("conv.weight", "conv.conv.weight")
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, "%s" % new_key
            model_dict[new_key] = state_dict[key]
        super(OFAResNets32x32_50l, self).load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_max_net(self):
        self.set_active_subnet(
            d=max(self.depth_list),
            e=max(self.expand_ratio_list),
            w=len(self.width_mult_list) - 1,
        )

    def set_active_subnet(self, d=None, e=None, w=None, **kwargs):
        depth = val2list(d, len(ResNets32x32.BASE_DEPTH_LIST))
        expand_ratio = val2list(e, len(self.blocks))
        width_mult = val2list(w, len(ResNets32x32.BASE_DEPTH_LIST) + 1)

        for block, e in zip(self.blocks, expand_ratio):
            if e is not None:
                block.active_expand_ratio = e

        if width_mult[0] is not None:
            self.input_stem[0].active_out_channel = self.input_stem[0].out_channel_list[
                width_mult[0]
            ]
        #         if width_mult[1] is not None:
        #             self.input_stem[2].active_out_channel = self.input_stem[2].out_channel_list[width_mult[1]]

        #         if depth[0] is not None:
        #             self.input_stem_skipping = (depth[0] != max(self.depth_list))
        for stage_id, (block_idx, d, w) in enumerate(
            zip(self.grouped_block_index, depth, width_mult[1:])
        ):
            if d is not None:
                self.runtime_depth[stage_id] = max(self.depth_list) - d
            if w is not None:
                for idx in block_idx:
                    self.blocks[idx].active_out_channel = self.blocks[
                        idx
                    ].out_channel_list[w]

    def sample_active_subnet(self):
        # sample expand ratio
        expand_setting = []
        for block in self.blocks:
            expand_setting.append(random.choice(block.expand_ratio_list))

        # sample depth
        depth_setting = [random.choice([max(self.depth_list), min(self.depth_list)])]
        for stage_id in range(len(ResNets32x32.BASE_DEPTH_LIST)):
            depth_setting.append(random.choice(self.depth_list))

        # sample width_mult
        width_mult_setting = [
            random.choice(list(range(len(self.input_stem[0].out_channel_list)))),
            random.choice(list(range(len(self.input_stem[2].out_channel_list)))),
        ]
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            stage_first_block = self.blocks[block_idx[0]]
            width_mult_setting.append(
                random.choice(list(range(len(stage_first_block.out_channel_list))))
            )

        arch_config = {
            "d": depth_setting,
            "e": expand_setting,
            "w": width_mult_setting,
        }
        self.set_active_subnet(**arch_config)
        return arch_config

    def get_active_subnet(self, preserve_weight=True):
        input_stem = [self.input_stem[0].get_active_subnet(3, preserve_weight)]
        #         if self.input_stem_skipping <= 0:
        #             input_stem.append(ResidualBlock(
        #                 self.input_stem[1].conv.get_active_subnet(self.input_stem[0].active_out_channel, preserve_weight),
        #                 IdentityLayer(self.input_stem[0].active_out_channel, self.input_stem[0].active_out_channel)
        #             ))
        #         input_stem.append(self.input_stem[2].get_active_subnet(self.input_stem[0].active_out_channel, preserve_weight))
        input_channel = self.input_stem[0].active_out_channel

        blocks = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                blocks.append(
                    self.blocks[idx].get_active_subnet(input_channel, preserve_weight)
                )
                input_channel = self.blocks[idx].active_out_channel
        classifier = self.classifier.get_active_subnet(input_channel, preserve_weight)
        subnet = ResNets32x32(input_stem, blocks, classifier)

        subnet.set_bn_param(**self.get_bn_param())
        return subnet

    def get_active_net_config(self):
        input_stem_config = [self.input_stem[0].get_active_subnet_config(3)]
        #         if self.input_stem_skipping <= 0:
        #             input_stem_config.append({
        #                 'name': ResidualBlock.__name__,
        #                 'conv': self.input_stem[1].conv.get_active_subnet_config(self.input_stem[0].active_out_channel),
        #                 'shortcut': IdentityLayer(self.input_stem[0].active_out_channel, self.input_stem[0].active_out_channel).config,
        #             })
        #         input_stem_config.append(self.input_stem[2].get_active_subnet_config(self.input_stem[0].active_out_channel))
        input_channel = self.input_stem[0].active_out_channel

        blocks_config = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                blocks_config.append(
                    self.blocks[idx].get_active_subnet_config(input_channel)
                )
                input_channel = self.blocks[idx].active_out_channel
        classifier_config = self.classifier.get_active_subnet_config(input_channel)
        return {
            "name": OFAResNets32x32_50l.__name__,
            "bn": self.get_bn_param(),
            "input_stem": input_stem_config,
            "blocks": blocks_config,
            "classifier": classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks:
            block.re_organize_middle_weights(expand_ratio_stage)
