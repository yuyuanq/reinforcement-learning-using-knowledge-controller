import numpy as np

threshold = 0.01


class PendulumRule:
    def __init__(self):
        self.rule_dict = {0: [[self.s0_low, self.s1_low, self.s2_po, self.a0_right],
                              [self.s0_low, self.s1_low, self.s2_ne, self.a0_left],
                              [self.s0_high_right, self.s1_high_right, self.s2_po, self.a0_right_inverse],
                              [self.s0_high_right, self.s1_high_right, self.s2_ne, self.a0_left_inverse],
                              [self.s0_high_left, self.s1_high_left, self.s2_ne, self.a0_left_inverse],
                              [self.s0_high_left, self.s1_high_left, self.s2_po, self.a0_right_inverse]]}

    @staticmethod
    def s_any(x):
        return np.piecewise(x, [x], [lambda x: 1])

    @staticmethod
    def s0_low(x):
        return np.piecewise(x, [x <= -1, (-1 < x) & (x <= 0), x > 0],
                            [lambda x: 0, lambda x: 1, lambda x: 0])

    @staticmethod
    def s1_low(x):
        return np.piecewise(x, [x <= -1, (-1 < x) & (x <= 1), x > 1],
                            [lambda x: 0, lambda x: 1, lambda x: 0])

    @staticmethod
    def s2_ne(x):
        return np.piecewise(x, [x <= -0.1, (-0.1 < x) & (x <= 0), x > 0],
                            [lambda x: 1, lambda x: -1 / 0.1 * x, lambda x: 0])

    @staticmethod
    def s2_po(x):
        return np.piecewise(x, [x <= 0, (0 < x) & (x <= 0.1), x > 0.1],
                            [lambda x: 0, lambda x: 1 / 0.1 * x, lambda x: 1])

    @staticmethod
    def s0_high_right(x):
        return np.piecewise(x, [x <= 0, (0 < x) & (x <= threshold), x > threshold],
                            [lambda x: 0, lambda x: 1 / threshold * x, lambda x: 1])

    @staticmethod
    def s1_high_right(x):
        return np.piecewise(x, [x <= -1, (-1 < x) & (x <= -1 + threshold), (x > -1 + threshold) & (x < 0)],
                            [lambda x: 0, lambda x: (x + 1) / threshold, lambda x: 1, lambda x: 0])

    @staticmethod
    def s0_high_left(x):
        return np.piecewise(x, [x <= 0, (0 < x) & (x <= threshold), x > threshold],
                            [lambda x: 0, lambda x: 1 / threshold * x, lambda x: 1])

    @staticmethod
    def s1_high_left(x):
        return np.piecewise(x, [x < 0, (x > 0) & (x <= 1 - threshold), (1 - threshold < x) & (x <= 1), x > 1],
                            [lambda x: 0, lambda x: 1, lambda x: 1 - 1 / threshold * (x - 1 + threshold), lambda x: 0])

    @staticmethod
    def a0_left(x):
        return -2 * x

    @staticmethod
    def a0_right(x):
        return 2 * x

    @staticmethod
    def a0_right_inverse(x):
        return -x * 2
        # return torch.zeros_like(x)
        # return torch.max(torch.cat([-8 * x + 2, torch.zeros_like(x)], 1), keepdim=True, dim=1)

    @staticmethod
    def a0_left_inverse(x):
        return x * 2
        # return torch.zeros_like(x)
        # return torch.min(torch.cat([8 * x - 2, torch.zeros_like(x)], 1), keepdim=True, dim=1)
