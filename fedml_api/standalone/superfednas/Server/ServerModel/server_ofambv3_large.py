from fedml_api.standalone.superfednas.Server.ServerModel.base_server_model import (
    BaseServerModel,
)
from fedml_api.standalone.superfednas.Client.ClientModel.client_model import ClientModel
from fedml_api.standalone.superfednas.elastic_nn.ofa_mbv3_large_32x32 import (
    OFAMobileNetV3_32x32,
    MobileNetV3,
)
from ofa.utils import make_divisible, SEModule, MyNetwork

import numpy as np
from ofa.utils import val2list
import copy
import random


class ServerMobilenetV3Large_32x32(BaseServerModel):
    def __init__(
        self,
        init_params,
        sampling_method,
        num_cli_total,
        bn_gamma_zero_init=False,
        cli_subnet_track=None,
    ):
        self.bn_gamma_zero_init = bn_gamma_zero_init
        super(ServerMobilenetV3Large_32x32, self).__init__(
            init_params,
            sampling_method,
            num_cli_total,
            cli_subnet_track=cli_subnet_track,
        )

    def init_model(self, init_params):
        return OFAMobileNetV3_32x32(**init_params)

    def active_subnet_index(self):
        mapping_active_subnet_to_supernet = {
            "blocks": dict(),
            "channels": dict(),
            "se": dict(),
        }

        mapping_active_subnet_to_supernet["blocks"][0] = 0
        mapping_active_subnet_to_supernet["channels"][0] = (
            self.model.blocks[0].conv.out_channels,
            self.model.blocks[0].conv.out_channels,
        )

        input_channel = self.model.blocks[0].conv.out_channels

        subnet_block_idx = 1
        for stage_id, block_idx in enumerate(self.model.grouped_block_index):
            depth_param = self.model.runtime_depth[stage_id]
            active_idx = block_idx[: min(len(block_idx), depth_param)]
            for idx in active_idx:
                mapping_active_subnet_to_supernet["blocks"][subnet_block_idx] = idx
                mid_channel = self.model.blocks[idx].conv.active_middle_channel(
                    input_channel
                )
                # encoding for  layers: (active out channels, active middle channels, input channel)
                mapping_active_subnet_to_supernet["channels"][subnet_block_idx] = (
                    self.model.blocks[idx].conv.active_out_channel,
                    mid_channel,
                    input_channel,
                )

                if self.model.blocks[idx].conv.use_se:
                    # encoding for SE layers: (se_mid, active middle channels)
                    se_mid = make_divisible(
                        mid_channel // SEModule.REDUCTION,
                        divisor=MyNetwork.CHANNEL_DIVISIBLE,
                    )
                    mapping_active_subnet_to_supernet["se"][subnet_block_idx] = (
                        se_mid,
                        mid_channel,
                    )

                input_channel = self.model.blocks[idx].conv.active_out_channel
                subnet_block_idx += 1
        return mapping_active_subnet_to_supernet

    def get_subnet(self, d=None, e=None, ks=None, preserve_weight=True, **kwargs):
        self.model.set_active_subnet(d=d, e=e, ks=ks, **kwargs)
        subnet = self.model.get_active_subnet(preserve_weight=preserve_weight)
        subindex = self.active_subnet_index()
        arch_config = dict()
        ks = val2list(ks, len(self.model.blocks) - 1)
        expand_ratio = val2list(e, len(self.model.blocks) - 1)
        depth = val2list(d, len(self.model.block_group_info))
        arch_config["d"] = depth
        arch_config["e"] = expand_ratio
        arch_config["ks"] = ks
        new_model = ClientModel(
            subnet, subindex, arch_config, self.is_max_net, self.random_subnet_arch, self.random_depth_subnet_arch()
        )
        self.model.set_max_net()
        return new_model

    def add_subnet(
        self, shared_param_sum, shared_param_count, w_local,
    ):
        weight = w_local.avg_weight
        local_model_params = w_local.state_dict()
        local_model_index = w_local.model_index
        for key in local_model_params:
            if "blocks" in key:
                delim = "."
                split_key = key.split(delim)
                layer_type, ind = split_key[0], int(split_key[1])
                split_key[1] = str(local_model_index[split_key[0]][int(split_key[1])])
                new_key = delim.join(split_key)
            else:
                new_key = key

            if ".mobile_inverted_conv." in new_key:
                new_key = new_key.replace(".mobile_inverted_conv.", ".conv.")

            if new_key in shared_param_sum:
                pass
            elif ".bn.bn." in new_key:
                new_key = new_key.replace(".bn.bn.", ".bn.")
            elif ".conv.conv.weight" in new_key:
                new_key = new_key.replace(".conv.conv.weight", ".conv.weight")
            elif ".linear.linear." in new_key:
                new_key = new_key.replace(".linear.linear.", ".linear.")
            ##############################################################################
            elif ".linear." in new_key:
                new_key = new_key.replace(".linear.", ".linear.linear.")
            elif "bn." in new_key:
                new_key = new_key.replace("bn.", "bn.bn.")
            elif "conv.weight" in new_key:
                new_key = new_key.replace("conv.weight", "conv.conv.weight")
            else:
                raise ValueError(new_key)
            assert new_key in shared_param_sum, "%s" % new_key

            if "conv.weight" in key:
                active_channels_in = None
                active_channels_out = None
                if "blocks" in key:

                    if "inverted_bottleneck" in key:
                        active_channels_out = local_model_index["channels"][ind][1]
                        active_channels_in = local_model_index["channels"][ind][2]
                    elif "depth_conv" in key:
                        active_channels_out = local_model_index["channels"][ind][1]
                        active_channels_in = local_model_index["channels"][ind][1]
                    elif "point_linear" in key:
                        active_channels_out = local_model_index["channels"][ind][0]
                        active_channels_in = local_model_index["channels"][ind][1]
                    else:
                        raise ValueError(f"Wrong layer: {key}")
                else:
                    conv_size = local_model_params[key].size()
                    active_channels_out = conv_size[0]
                    active_channels_in = conv_size[1]

                try:
                    shared_param_sum[new_key][
                        :active_channels_out, :active_channels_in, :, :
                    ] += (weight * local_model_params[key])
                    shared_param_count[new_key][
                        :active_channels_out, :active_channels_in, :, :
                    ] += weight
                except Exception as e:
                    raise ValueError(f"Wrong layer: {key} exception:  {e}")

            elif "se" in key:
                assert "blocks" in key
                if "bias" not in key:
                    active_channels_out, active_channels_in = None, None
                    if "reduce" in key:
                        active_channels_out = local_model_index["se"][ind][0]
                        active_channels_in = local_model_index["se"][ind][1]
                    elif "expand" in key:
                        active_channels_out = local_model_index["se"][ind][1]
                        active_channels_in = local_model_index["se"][ind][0]
                    else:
                        raise ValueError(f"Wrong layer: {key}")
                    shared_param_sum[new_key][
                        :active_channels_out, :active_channels_in, :, :
                    ] += (weight * local_model_params[key])
                    shared_param_count[new_key][
                        :active_channels_out, :active_channels_in, :, :
                    ] += weight
                else:
                    active_channel = local_model_params[key].size()[0]
                    shared_param_sum[new_key][:active_channel] += (
                        weight * local_model_params[key]
                    )
                    shared_param_count[new_key][:active_channel] += weight
            elif "bn" in key:
                if len(local_model_params[key].shape) > 0:
                    shared_param_sum[new_key][: local_model_params[key].shape[0]] += (
                        weight * local_model_params[key]
                    )
                    shared_param_count[new_key][
                        : local_model_params[key].shape[0]
                    ] += weight
            elif "linear" in key:
                shared_param_sum[new_key] += weight * local_model_params[key]
                shared_param_count[new_key] += weight
            else:
                raise ValueError(f"Wrong layer: {key}")

    def is_max_net(self, arch):
        largest = True
        max_e = max(self.model.expand_ratio_list)
        max_d = max(self.model.depth_list)
        max_ks = max(self.model.ks_list)
        for d in arch["d"]:
            if d != max_d:
                largest = False
        for e in arch["e"]:
            if e != max_e:
                largest = False
        if "ks" in arch:
            for ks in arch["ks"]:
                if ks != max_ks:
                    largest = False
        return largest

    def is_min_net(self, arch):
        smallest = True
        min_e = min(self.model.expand_ratio_list)
        min_d = min(self.model.depth_list)
        min_ks = min(self.model.ks_list)
        for d in arch["d"]:
            if d != min_d:
                smallest = False
        for e in arch["e"]:
            if e != min_e:
                smallest = False
        if "ks" in arch:
            for ks in arch["ks"]:
                if ks != min_ks:
                    smallest = False
        return smallest

    def max_subnet_arch(self):
        max_e = max(self.model.expand_ratio_list)
        max_d = max(self.model.depth_list)
        max_ks = max(self.model.ks_list)
        expand_setting, depth_setting, ks_setting = [], [], []
        for stage_id, block_idx in enumerate(self.model.grouped_block_index):
            for _ in block_idx:
                expand_setting.append(max_e)
                ks_setting.append(max_ks)
            depth_setting.append(max_d)

        return {"d": depth_setting, "e": expand_setting, "ks": ks_setting}

    def min_subnet_arch(self):
        min_e = min(self.model.expand_ratio_list)
        min_d = min(self.model.depth_list)
        min_ks = min(self.model.ks_list)
        expand_setting, depth_setting, ks_setting = [], [], []
        for stage_id, block_idx in enumerate(self.model.grouped_block_index):
            for _ in block_idx:
                expand_setting.append(min_e)
                ks_setting.append(min_ks)
            depth_setting.append(min_d)

        return {"d": depth_setting, "e": expand_setting, "ks": ks_setting}

    def random_subnet_arch(self):
        expand_setting, depth_setting, ks_setting = [], [], []
        for stage_id, block_idx in enumerate(self.model.grouped_block_index):
            for _ in block_idx:
                expand_setting.append(np.random.choice(self.model.expand_ratio_list))
                ks_setting.append(int(np.random.choice(self.model.ks_list)))

            depth_setting.append(np.random.choice(self.model.depth_list))
        return {"d": depth_setting, "e": expand_setting, "ks": ks_setting}

    def random_depth_subnet_arch(self):
        expand_setting, depth_setting, ks_setting = [], [], []
        for stage_id, block_idx in enumerate(self.model.grouped_block_index):
            for _ in block_idx:
                expand_setting.append(max(self.model.expand_ratio_list))
                ks_setting.append(int(max(self.model.ks_list)))

            depth_setting.append(np.random.choice(self.model.depth_list))
        return {"d": depth_setting, "e": expand_setting, "ks": ks_setting}

    def random_compound_subnet_arch(self):
        def clip_expands(expands):
            low = min(self.expand_ratio_list)
            expands = list(set(np.clip(expands, low, None)))
            return expands

        mapping = {
            2: clip_expands([3]),
            3: clip_expands([4]),
            4: clip_expands([6]),
        }

        expand_setting, depth_setting, ks_setting = [], [], []
        for stage_id, block_idx in enumerate(self.model.grouped_block_index):
            chosen_depth = np.random.choice(self.model.depth_list)
            depth_setting.append(chosen_depth)
            for _ in block_idx:
                expand_setting.append(np.random.choice(mapping[chosen_depth]))
                ks_setting.append(int(np.random.choice(self.model.ks_list)))

        return {"d": depth_setting, "e": expand_setting, "ks": ks_setting}

    def mutate_sample(self, sample_arch, mut_prob):
        new_sample = copy.deepcopy(sample_arch)
        depth_list = self.model.depth_list
        expand_ratio_list = self.model.expand_ratio_list

        for stage_id, block_idx in enumerate(self.model.grouped_block_index):
            if random.random() < mut_prob:
                new_sample["d"][stage_id] = np.random.choice(depth_list)
            for idx in block_idx:
                if random.random() < mut_prob:
                    new_sample["e"][idx] = np.random.choice(expand_ratio_list)
                if random.random() < mut_prob:
                    new_sample["ks"][idx] = int(np.random.choice(self.model.ks_list))
        return new_sample
