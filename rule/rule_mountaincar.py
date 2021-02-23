import numpy as np


class MountainCarRule:
    def __init__(self):
        self.rule_dict = {0: [[self.s_any, self.s1_ne]],
                          1: [[self.s_no, self.s_no]],
                          2: [[self.s_any, self.s1_po]]}

    @staticmethod
    def s_any(x):
        return np.piecewise(x, [x], [lambda x: 1])

    @staticmethod
    def s_no(x):
        return np.piecewise(x, [x], [lambda x: 0])

    @staticmethod
    def s1_ne(x):
        return np.piecewise(x, [x <= -0.01, (-0.01 < x) & (x <= 0), x > 0],
                            [lambda x: 1, lambda x: -100 * x, lambda x: 0])

    @staticmethod
    def s1_po(x):
        return np.piecewise(x, [x <= 0, (0 < x) & (x <= 0.01), x > 0.01],
                            [lambda x: 0, lambda x: 100 * x, lambda x: 1])
