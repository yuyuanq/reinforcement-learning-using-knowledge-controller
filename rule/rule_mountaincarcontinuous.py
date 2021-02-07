import numpy as np


class MountainCarContinuousRule:
    def __init__(self):
        self.rule_dict = {0: [[self.s_any, self.s1_ne, self.a0_left],
                              [self.s_any, self.s1_po, self.a0_right]]}

    @staticmethod
    def s_any(x):
        return np.piecewise(x, [x], [lambda x: 1])

    @staticmethod
    def s1_ne(x):
        return np.piecewise(x, [x <= -0.01, (-0.01 < x) & (x <= 0), x > 0],
                            [lambda x: 1, lambda x: -100 * x, lambda x: 0])

    @staticmethod
    def s1_po(x):
        return np.piecewise(x, [x <= 0, (0 < x) & (x <= 0.01), x > 0.01],
                            [lambda x: 0, lambda x: 100 * x, lambda x: 1])

    @staticmethod
    def a0_left(x):
        return -x

    @staticmethod
    def a0_right(x):
        return x
