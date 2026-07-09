from abc import ABC, abstractmethod


class LRScheduler(ABC):
    """
    Abstract base class for FLOFA learning rate scheduler.
    """

    def __init__(self, num_diverse_subnets, args=None):
        self.num_diverse_subnets = num_diverse_subnets
        self.args = args

    @abstractmethod
    def get_flofa_lrs(self, round_num):
        pass

    def get_subnet_flofa_lr(self, round_num, subnet_idx):
        return self.get_flofa_lrs(round_num)[subnet_idx]
