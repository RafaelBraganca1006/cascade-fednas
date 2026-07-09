from fedml_api.standalone.superfednas.learning_rate_schedule.superfednas_lr_scheduler import (
    LRScheduler,
)


class LinearLRSchedule(LRScheduler):
    def __init__(
        self,
        num_diverse_subnets,
        initial_flofa_lrs,
        final_flofa_lr,
        largest_subnet_id,
        peak_rounds,
        end_R,
        args=None,
    ):
        super(LinearLRSchedule, self).__init__(num_diverse_subnets, args)
        self.initial_flofa_lrs = initial_flofa_lrs
        self.final_flofa_lr = final_flofa_lr
        self.largest_subnet_id = largest_subnet_id
        self.peak_rounds = peak_rounds
        self.end_R = end_R

    def get_flofa_lrs(self, round_num):
        cur_lrs = [None] * self.num_diverse_subnets
        for id in range(self.num_diverse_subnets):
            cur_lrs[id] = self.get_subnet_flofa_lr(round_num, id)
        return cur_lrs

    def largest_subnet_lr(self, round_num):
        diff = self.final_flofa_lr - self.initial_flofa_lrs[self.largest_subnet_id]
        increment = diff / self.end_R
        return self.initial_flofa_lrs[self.largest_subnet_id] + increment * min(
            round_num, self.end_R
        )

    def get_subnet_flofa_lr(self, round_num, id):
        if id == self.largest_subnet_id or round_num > self.peak_rounds[id]:
            return self.largest_subnet_lr(round_num)
        peak_R = self.peak_rounds[id]
        diff = self.largest_subnet_lr(peak_R) - self.initial_flofa_lrs[id]
        increment = diff / peak_R
        return self.initial_flofa_lrs[id] + increment * round_num
