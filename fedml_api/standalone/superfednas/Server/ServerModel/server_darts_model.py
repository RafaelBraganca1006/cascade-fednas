from fedml_api.standalone.superfednas.Client.ClientModel import ClientModel
from fedml_api.standalone.superfednas.Server.ServerModel.base_server_model import (
    BaseServerModel,
)
from fedml_api.standalone.superfednas.elastic_nn.darts import genotypes
from fedml_api.standalone.superfednas.elastic_nn.darts.model import NetworkCIFAR
import torch

class ServerDarts(BaseServerModel):
    def __init__(
        self,
        init_params,
        sampling_method,
        num_cli_total,
        cli_subnet_track=None,
    ):
        super(ServerDarts, self).__init__(
            init_params,
            sampling_method,
            num_cli_total,
            cli_subnet_track=cli_subnet_track,
        )

    def init_model(self, init_params):
        return NetworkCIFAR(**init_params)

    def active_subnet_index(self):
        return genotypes.FedNAS_V1

    def get_subnet(self, d=None, e=None, w=None, **kwargs):
        #is a single architecture
        new_model = ClientModel(
            self.model, genotypes.FedNAS_V1, genotypes.FedNAS_V1, self.is_max_net, self.random_subnet_arch, self.random_depth_subnet_arch()
        )
        return new_model

    def add_subnet(
        self, shared_param_sum, shared_param_count, w_local,
    ):
        weight = w_local.avg_weight
        local_model_params = w_local.state_dict()
        for key in local_model_params:
            shared_param_sum[key] += weight * local_model_params[key]
            shared_param_count[key] += weight

    def is_max_net(self, arch):
        return True

    def is_min_net(self, arch):
        return True

    def max_subnet_arch(self):
        return genotypes.FedNAS_V1

    def min_subnet_arch(self):
        return genotypes.FedNAS_V1

    def random_subnet_arch(self):
        return genotypes.FedNAS_V1

    def random_depth_subnet_arch(self):
        return genotypes.FedNAS_V1

    def random_compound_subnet_arch(self):
        raise Exception("Random compound subnet sampling not supported by ETCN!")

    def mutate_sample(self, sample_arch, mut_prob):
        return genotypes.FedNAS_V1

    def wandb_pass(self):
        self.model.eval()
        self.model(torch.zeros((1, 3, 32, 32)))
