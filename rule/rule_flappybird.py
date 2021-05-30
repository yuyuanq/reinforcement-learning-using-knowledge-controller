import numpy as np


class FlappyBirdRule:
    def __init__(self):
        self.rule_dict = {
            0: [[self.s0_sm, self.s_any, self.s_any, self.s_any, self.s_any],
                [self.s_any, self.s1_ne, self.s_any, self.s3_ne, self.s_any]],
            1: [[self.s0_la, self.s1_po, self.s_any, self.s_any, self.s_any],
                [self.s_any, self.s1_po, self.s_any, self.s_any, self.s4_po]]}

    @staticmethod
    def s_any(x):
        return np.piecewise(x, [x], [lambda x: 1])

    @staticmethod
    def s0_la(x):
        return np.piecewise(x, [x > 2.5, (2 < x) & (x <= 2.5), x <= 2],
                            [lambda x: 1, lambda x: (1 / 0.5) * x - 4, lambda x: 0])

    @staticmethod
    def s0_sm(x):
        return np.piecewise(x, [x > 2, (1.5 < x) & (x <= 2), x <= 1.5],
                            [lambda x: 0, lambda x: -(1 / 0.5) * x + 4, lambda x: 1])

    @staticmethod
    def s1_po(x):
        return np.piecewise(x, [x > 0.06, (0 < x) & (x <= 0.06), x <= 0],
                            [lambda x: 1, lambda x: (1 / 0.06) * x, lambda x: 0])

    @staticmethod
    def s1_ne(x):
        return np.piecewise(x, [x > 0, (-0.06 < x) & (x <= 0), x <= -0.06],
                            [lambda x: 0, lambda x: -(1 / 0.06) * x, lambda x: 1])

    @staticmethod
    def s3_ne(x):
        return np.piecewise(x, [x > 0, (-0.3 < x) & (x <= 0), x <= -0.3],
                            [lambda x: 0, lambda x: -(1 / 0.3) * x, lambda x: 1])

    @staticmethod
    def s4_po(x):
        return np.piecewise(x, [x > 0.3, (0 < x) & (x <= 0.3), x <= 0],
                            [lambda x: 1, lambda x: (1 / 0.3) * x, lambda x: 0])
