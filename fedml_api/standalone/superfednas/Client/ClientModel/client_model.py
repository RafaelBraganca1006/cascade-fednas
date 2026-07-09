class ClientModel:
    def __init__(
        self,
        model,
        model_index,
        model_config,
        is_max_net,
        sample_random_subnet=None,
        sample_random_depth_subnet=None,
        avg_weight=1,
    ):
        self.model = model
        self.model_index = model_index
        self.model_config = model_config
        self.is_max_net = is_max_net
        self.avg_weight = avg_weight
        self.sample_random_subnet = sample_random_subnet
        self.sample_random_depth_subnet = sample_random_depth_subnet

    def get_model(self):
        return self.model

    def state_dict(self):
        return self.model.cpu().state_dict()

    def to(self, device):
        self.model.to(device)

    def cpu(self):
        self.model = self.model.cpu()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def modules(self):
        return self.model.modules()

    def parameters(self):
        return self.model.parameters()

    def zero_grad(self):
        self.model.zero_grad()

    def forward(self, x):
        return self.model(x)

    def set_avg_wt(self, a):
        self.avg_weight = a

    def freeze(self):
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.eval()

    def set_active_subnet(self, arch):
        self.model.set_active_subnet(**arch)

    def set_max_net(self):
        self.model.set_max_net()
