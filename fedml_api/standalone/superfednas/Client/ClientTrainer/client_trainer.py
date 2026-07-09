from abc import ABC, abstractmethod
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
import torch
from torch import nn


class ClientTrainer(ABC):
    def __init__(self, model, device, args, teacher_model=None):
        self.client_model = model
        self.test_model = None
        self.device = device
        self.args = args
        self.client_idx = None
        self.local_training_data = None
        self.local_test_data = None
        self.local_sample_number = None
        self.seed = 0
        self.teacher_model = teacher_model

    def cross_entropy_loss_with_soft_target(self, pred, soft_target):
        logsoftmax = nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))

    def get_sample_number(self):
        return self.local_sample_number

    def update_local_dataset(
        self, client_idx, local_training_data, local_test_data, local_sample_number,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def set_model(self, model):
        self.client_model = model

    def set_test_model(self, model):
        self.test_model = model

    def local_test(self, use_test_set):
        if use_test_set:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data

        if self.args.reset_bn_stats or self.args.reset_bn_stats_test:
            data_loader = self.random_sub_train_loader(
                int(self.args.reset_bn_sample_size * len(self.local_training_data)),
                self.args.batch_size,
            )
            set_running_statistics(self.client_model.get_model(), data_loader)
        metrics = self.test(test_data, self.args)
        return metrics

    def random_sub_train_loader(self, subset_size, subset_batch_size):
        n_samples = len(self.local_training_data)
        g = torch.Generator()
        g.manual_seed(self.seed)
        self.seed += 1
        rand_indexes = torch.randperm(n_samples, generator=g).tolist()
        sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            rand_indexes[:subset_size]
        )
        subset = []
        sub_data_loader = torch.utils.data.DataLoader(
            self.local_training_data.dataset,
            batch_size=subset_batch_size,
            sampler=sub_sampler,
            pin_memory=True,
        )
        for images, labels in sub_data_loader:
            subset.append((images, labels))
        return subset

    @abstractmethod
    def test(self, dataset, device, args, **kwargs):
        pass

    @abstractmethod
    def train(self, lr, local_ep, **kwargs):
        pass
