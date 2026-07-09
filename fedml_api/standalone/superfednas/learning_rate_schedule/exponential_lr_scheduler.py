class ExponentialLRSchedule:
    def __init__(
        self, init_lr, decay_rate, decay_freq,
    ):
        self.init_lr = init_lr
        self.base = 1/float(decay_rate)
        self.decay_freq = decay_freq

    def get_lr(self, round_num):
        exp = int(round_num/self.decay_freq)
        return self.init_lr*(self.base**exp)
