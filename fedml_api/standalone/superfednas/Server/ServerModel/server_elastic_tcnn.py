from fedml_api.standalone.superfednas.Server.ServerModel.base_server_model import (
    BaseServerModel,
)
from fedml_api.standalone.superfednas.Client.ClientModel.client_model import ClientModel

from fedml_api.standalone.superfednas.elastic_nn.TCN.elastic_tcnn import (
    ElasticTCN
)

import torch.nn as nn
import numpy as np
import random
import copy
from ofa.utils import val2list


class ServerElasticTCNN(BaseServerModel):
    def __init__(
        self,
        init_params,
        sampling_method,
        num_cli_total,
        cli_subnet_track=None,
    ):
        super(ServerElasticTCNN, self).__init__(
            init_params,
            sampling_method,
            num_cli_total,
            cli_subnet_track=cli_subnet_track,
        )

    def init_model(self, init_params):
        return ElasticTCN(**init_params)

    def active_subnet_index(self):
        mapping_active_subnet_to_supernet = {
            "depth": len(self.model.tcn.blocks) - self.model.tcn.runtime_depth,
            "middle": [],
            "in": [],
            "out": []
        }
        input_channel = self.model.tcn.input_channel
        for ind, block in enumerate(self.model.tcn.blocks):
            mapping_active_subnet_to_supernet["middle"].append(block.active_middle_channels)
            mapping_active_subnet_to_supernet["out"].append(block.active_out_channel)
            mapping_active_subnet_to_supernet["in"].append(input_channel)
            input_channel = block.active_out_channel
        return mapping_active_subnet_to_supernet

    def get_subnet(self, d=None, e=None, w=None, **kwargs):
        self.model.set_active_subnet(d=d, e=e, w=w, **kwargs)
        subnet = self.model.get_active_subnet()
        subindex = self.active_subnet_index()
        arch_config = dict()
        depth = val2list(d, len(self.model.tcn.depth_list))
        expand_ratio = val2list(e, len(self.model.tcn.expand_ratio_list))
        arch_config["d"] = depth
        arch_config["e"] = expand_ratio
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
            if "tcn" in key:
                delim = "."
                split_key = key.split(delim)
                block_idx = int(split_key[2])
                conv_id = split_key[3]
                in_size = local_model_index["in"][block_idx]
                out = local_model_index["out"][block_idx]
                middle = local_model_index["middle"][block_idx]
                if "weight_g" in key:
                    if conv_id == "conv1":
                        shared_param_sum[key][:middle, :, :] += weight*local_model_params[key]
                        shared_param_count[key][:middle, :, :] += weight
                    elif conv_id == "conv2":
                        shared_param_sum[key][:out, :, :] += weight*local_model_params[key]
                        shared_param_count[key][:out, :, :] += weight
                    else:
                        raise Exception("Unexpected parameter found!")

                elif "weight_v" in key:
                    if conv_id == "conv1":
                        shared_param_sum[key][:middle, :in_size, :] += weight*local_model_params[key]
                        shared_param_count[key][:middle, :in_size, :] += weight
                    elif conv_id == "conv2":
                        shared_param_sum[key][:out, :middle, :] += weight*local_model_params[key]
                        shared_param_count[key][:out, :middle, :] += weight
                    else:
                        raise Exception("Unexpected parameter found!")
                elif "bias" in key:
                    if conv_id == "conv1":
                        shared_param_sum[key][:middle] += weight*local_model_params[key]
                        shared_param_count[key][:middle] += weight
                    elif conv_id == "conv2":
                        shared_param_sum[key][:out] += weight*local_model_params[key]
                        shared_param_count[key][:out] += weight
                    else:
                        raise Exception("Unexpected parameter found!")
                else:
                    raise Exception("Unexpected parameter found!")
            else:
                shared_param_sum[key] += weight * local_model_params[key]
                shared_param_count[key] += weight

    def is_max_net(self, arch):
        largest = True
        max_depth = max(self.model.tcn.depth_list)
        max_expand_ratio = max(self.model.tcn.expand_ratio_list)
        if arch["d"] != max_depth:
            largest = False
        for e in arch["e"]:
            if e != max_expand_ratio:
                largest = False
        return largest

    def is_min_net(self, arch):
        smallest = True
        min_depth = min(self.model.tcn.depth_list)
        min_expand_ratio = min(self.model.tcn.expand_ratio_list)
        if arch["d"] != min_depth:
            smallest = False

        for e in arch["e"]:
            if e != min_expand_ratio:
                smallest = False
        return smallest

    def max_subnet_arch(self):
        max_depth = max(self.model.tcn.depth_list)
        max_expand_ratio = max(self.model.tcn.expand_ratio_list)
        return {"d": max_depth, "e": max_expand_ratio}

    def min_subnet_arch(self):
        min_depth = min(self.model.tcn.depth_list)
        min_expand_ratio = min(self.model.tcn.expand_ratio_list)
        return {"d": min_depth, "e": min_expand_ratio}

    def random_subnet_arch(self):
        # fixed depth_list
        depth_list = self.model.tcn.depth_list
        # sample expand ratio
        expand_setting = []
        for block in self.model.tcn.blocks:
            expand_setting.append(random.choice(block.expand_ratio_list))
        # sample depth
        depth_setting = np.random.choice(depth_list)

        arch_config = {
            "d": depth_setting,
            "e": expand_setting,
        }
        return arch_config

    def random_depth_subnet_arch(self):
        # fixed depth_list
        depth_list = self.model.tcn.depth_list
        # sample expand ratio
        expand_setting = []
        for block in self.model.tcn.blocks:
            expand_setting.append(max(block.expand_ratio_list))
        # sample depth
        depth_setting = np.random.choice(depth_list)

        arch_config = {
            "d": depth_setting,
            "e": expand_setting,
        }
        return arch_config

    def random_compound_subnet_arch(self):
        raise Exception("Random compound subnet sampling not supported by ETCN!")

    def mutate_sample(self, sample_arch, mut_prob):
        new_sample = copy.deepcopy(sample_arch)
        depth_list = self.model.tcn.depth_list

        # Resample expand ratios
        for i in range(len(self.model.tcn.blocks)):
            if random.random() < mut_prob:
                new_sample["e"][i] = random.choice(self.model.tcn.blocks[i].expand_ratio_list)

        # Resample depth
        if random.random() < mut_prob:
            new_sample["d"] = np.random.choice(depth_list)
        return new_sample
