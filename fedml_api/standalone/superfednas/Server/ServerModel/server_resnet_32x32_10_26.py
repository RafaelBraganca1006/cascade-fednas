from fedml_api.standalone.superfednas.Server.ServerModel.base_server_model import (
    BaseServerModel,
)
from fedml_api.standalone.superfednas.Client.ClientModel.client_model import ClientModel
from fedml_api.standalone.superfednas.elastic_nn.ofa_resnets_32x32_10_26 import (
    OFAResNets32x32_10_26,
    ResNets32x32_10_26,
)
import numpy as np
from ofa.utils import val2list
import random
import copy


class ServerResnet_10_26(BaseServerModel):
    def __init__(
        self,
        init_params,
        sampling_method,
        num_cli_total,
        bn_gamma_zero_init=False,
        cli_subnet_track=None,
    ):
        self.bn_gamma_zero_init = bn_gamma_zero_init
        super(ServerResnet_10_26, self).__init__(
            init_params,
            sampling_method,
            num_cli_total,
            cli_subnet_track=cli_subnet_track,
        )

    def init_model(self, init_params):
        return OFAResNets32x32_10_26(
            **init_params, bn_gamma_zero_init=self.bn_gamma_zero_init
        )

    def active_subnet_index(self):
        mapping_active_subnet_to_supernet = {
            "input_stem": dict(),
            "stem_channels": dict(),
            "blocks": dict(),
            "channels": dict(),
        }
        # Channel encoding: (Number of output channels, number of input channels)
        mapping_active_subnet_to_supernet["input_stem"][0] = 0
        mapping_active_subnet_to_supernet["stem_channels"][0] = (
            self.model.input_stem[0].active_out_channel,
            3,
        )

        input_channel = self.model.input_stem[0].active_out_channel

        subnet_block_idx = 0
        for stage_id, block_idx in enumerate(self.model.grouped_block_index):
            depth_param = self.model.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                mapping_active_subnet_to_supernet["blocks"][subnet_block_idx] = idx
                # encoding for resnet bottleneck layers: (active out channels, active middle channels, input channel)
                mapping_active_subnet_to_supernet["channels"][subnet_block_idx] = (
                    self.model.blocks[idx].active_out_channel,
                    self.model.blocks[idx].active_middle_channels,
                    input_channel,
                )
                input_channel = self.model.blocks[idx].active_out_channel
                subnet_block_idx += 1
        return mapping_active_subnet_to_supernet

    def get_subnet(self, d=None, e=None, w=None, preserve_weight=True, **kwargs):
        self.model.set_active_subnet(d=d, e=e, w=w, **kwargs)
        subnet = self.model.get_active_subnet(preserve_weight=preserve_weight)
        subindex = self.active_subnet_index()
        arch_config = dict()
        depth = val2list(d, len(ResNets32x32_10_26.BASE_DEPTH_LIST))
        expand_ratio = val2list(e, len(self.model.blocks))
        width_mult = val2list(w, len(ResNets32x32_10_26.BASE_DEPTH_LIST) + 1)
        arch_config["d"] = depth
        arch_config["e"] = expand_ratio
        arch_config["w"] = width_mult
        new_model = ClientModel(
            subnet, subindex, arch_config, self.is_max_net, self.random_subnet_arch, self.random_depth_subnet_arch,
        )
        self.model.set_max_net()
        return new_model

    def add_subnet(self, shared_param_sum, shared_param_count, w_local):
        weight = w_local.avg_weight
        local_model_params = w_local.state_dict()
        local_model_index = w_local.model_index
        classifier_active_in = None
        for key in local_model_params:
            if "classifier" not in key:
                delim = "."
                split_key = key.split(delim)
                layer_type, ind = split_key[0], int(split_key[1])
                split_key[1] = str(local_model_index[split_key[0]][int(split_key[1])])
                new_key = delim.join(split_key)
            else:
                new_key = key

            if new_key in shared_param_sum.keys():
                pass
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
                # Sum across active channels of convolution layers
                active_channels_in = None
                active_channels_out = None
                if layer_type == "input_stem":
                    active_channels_in = local_model_index["stem_channels"][ind][1]
                    active_channels_out = local_model_index["stem_channels"][ind][0]
                elif layer_type == "blocks":

                    if "conv1" in key:
                        active_channels_in = local_model_index["channels"][ind][2]
                        active_channels_out = local_model_index["channels"][ind][1]
                    elif "conv2" in key:
                        active_channels_in = local_model_index["channels"][ind][1]
                        active_channels_out = local_model_index["channels"][ind][0]
                        classifier_active_in = local_model_index["channels"][ind][0]

                    if "downsample" in key:
                        active_channels_in = local_model_index["channels"][ind][2]
                        active_channels_out = local_model_index["channels"][ind][0]
                shared_param_sum[new_key][
                    :active_channels_out, :active_channels_in, :, :
                ] += (weight * local_model_params[key])
                shared_param_count[new_key][
                    :active_channels_out, :active_channels_in, :, :
                ] += weight
            elif "linear.weight" in key:
                # Sum only across the active in channels to classifier
                shared_param_sum[new_key][:, :classifier_active_in] += (
                    weight * local_model_params[key]
                )
                shared_param_count[new_key][:, :classifier_active_in] += weight
            elif "bn" in key:
                if len(local_model_params[key].shape) > 0:
                    shared_param_sum[new_key][: local_model_params[key].shape[0]] += (
                        weight * local_model_params[key]
                    )
                    shared_param_count[new_key][
                        : local_model_params[key].shape[0]
                    ] += weight
            else:
                shared_param_sum[new_key] += weight * local_model_params[key]
                shared_param_count[new_key] += weight

    def is_max_net(self, arch):
        largest = True
        for d in arch["d"]:
            if d != 2:
                largest = False
        for e in arch["e"]:
            if e != 0.25:
                largest = False
        return largest

    def is_min_net(self, arch):
        smallest = True
        for d in arch["d"]:
            if d != 0:
                smallest = False

        for e in arch["e"]:
            if e != 0.1:
                smallest = False
        return smallest

    def max_subnet_arch(self):
        expand_setting = []
        for stage_id, block_idx in enumerate(self.model.grouped_block_index):
            for _ in block_idx:
                expand_setting.append(0.25)
        return {"d": [2, 2, 2, 2], "e": expand_setting}

    def min_subnet_arch(self):
        expand_setting = []
        for stage_id, block_idx in enumerate(self.model.grouped_block_index):
            for _ in block_idx:
                expand_setting.append(0.1)
        return {"d": [0, 0, 0, 0], "e": expand_setting}

    def random_subnet_arch(self):
        # fixed depth_list
        depth_list = [0, 1, 2]
        # fixed expand_ratio_list
        expand_ratio_list = [0.1, 0.14, 0.18, 0.22, 0.25]
        # sample expand ratio
        expand_setting = []
        for _ in self.model.blocks:
            expand_setting.append(np.random.choice(expand_ratio_list))

        # sample depth
        depth_setting = []
        for stage_id in range(len(ResNets32x32_10_26.BASE_DEPTH_LIST)):
            depth_setting.append(np.random.choice(depth_list))

        arch_config = {
            "d": depth_setting,
            "e": expand_setting,
        }
        return arch_config

    def random_depth_subnet_arch(self):
        # fixed depth_list
        depth_list = [0, 1, 2]
        # fixed expand_ratio_list
        expand_ratio_list = [0.1, 0.14, 0.18, 0.22, 0.25]
        # sample expand ratio
        expand_setting = []
        for _ in self.model.blocks:
            expand_setting.append(max(expand_ratio_list))

        # sample depth
        depth_setting = []
        for stage_id in range(len(ResNets32x32_10_26.BASE_DEPTH_LIST)):
            depth_setting.append(np.random.choice(depth_list))

        arch_config = {
            "d": depth_setting,
            "e": expand_setting,
        }
        return arch_config

    def random_compound_subnet_arch(self):
        # fixed depth_list
        depth_list = [0, 1, 2]
        # fixed expand_ratio_list mapping
        exp_ratio_map = {0: [0.1], 1: [0.14, 0.18], 2: [0.22, 0.25]}
        # sample depth
        depth_setting = []
        for stage_id in range(len(ResNets32x32_10_26.BASE_DEPTH_LIST)):
            depth_setting.append(np.random.choice(depth_list))

        # sample expand ratio
        expand_setting = []
        for stage_id, (block_idx, d) in enumerate(
            zip(self.model.grouped_block_index, depth_setting[1:])
        ):
            for _ in block_idx:
                expand_setting.append(np.random.choice(exp_ratio_map[d]))

        arch_config = {
            "d": depth_setting,
            "e": expand_setting,
        }
        return arch_config

    def mutate_sample(self, sample_arch, mut_prob):
        new_sample = copy.deepcopy(sample_arch)
        depth_list = [0, 1, 2]
        expand_ratio_list = [0.1, 0.14, 0.18, 0.22, 0.25]

        # Resample expand ratios
        for i in range(len(self.model.blocks)):
            if random.random() < mut_prob:
                new_sample["e"][i] = np.random.choice(expand_ratio_list)

        # Resample depth
        for i in range(len(ResNets32x32_10_26.BASE_DEPTH_LIST)):
            if random.random() < mut_prob:
                new_sample["d"][i] = np.random.choice(depth_list)

        return new_sample
