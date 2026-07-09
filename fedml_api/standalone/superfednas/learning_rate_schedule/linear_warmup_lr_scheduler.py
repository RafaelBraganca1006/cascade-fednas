class LinearWarmupLRSchedule:
    def __init__(
        self, init_lr, final_lr, numR,
    ):
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.numR = numR

    def get_lr(self, round_num):
        increment = float(self.final_lr - self.init_lr) / self.numR
        return self.init_lr + increment * min(round_num, self.numR)
