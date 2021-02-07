import numpy as np


class LunarLanderRule:
    def __init__(self):
        self.rule_dict = {
            0: [[self.s_no, self.s_no, self.s_no, self.s_no, self.s_no, self.s_no]],
            3: [[self.s_any, self.s_any, self.s_any, self.s_any, self.s4_po, self.s5_po],
                [self.s0_ne, self.s_any, self.s_any, self.s_any, self.s4_sm, self.s_any]],
            2: [[self.s_any, self.s1_sm, self.s_any, self.s3_la, self.s_any, self.s_any]],
            1: [[self.s_any, self.s_any, self.s_any, self.s_any, self.s4_ne, self.s5_ne],
                [self.s0_po, self.s_any, self.s_any, self.s_any, self.s4_sm, self.s_any]]}

    @staticmethod
    def s_any(x):
        return np.piecewise(x, [x], [lambda x: 1])

    @staticmethod
    def s_no(x):
        return np.piecewise(x, [x], [lambda x: 0])

    @staticmethod
    def s0_ne(x):
        return np.piecewise(x, [x > 0, (-0.05 < x) & (x <= 0), x <= -0.05],
                            [lambda x: 0, lambda x: -20 * x, lambda x: 1])

    @staticmethod
    def s0_po(x):
        return np.piecewise(x, [x > 0.05, (0 < x) & (x <= 0.05), x <= 0],
                            [lambda x: 1, lambda x: 20 * x, lambda x: 0])

    @staticmethod
    def s1_sm(x):
        return np.piecewise(x, [x > 0.7, (0 < x) & (x <= 0.7), x <= 0],
                            [lambda x: 0, lambda x: 1 - (1 / 0.7) * x, lambda x: 1])

    @staticmethod
    def s2_ne(x):
        return np.piecewise(x, [x > 0, (-0.3 < x) & (x <= 0), x <= -0.3],
                            [lambda x: 0, lambda x: -1 / 0.3 * x, lambda x: 1])

    @staticmethod
    def s2_po(x):
        return np.piecewise(x, [x > 0.3, (0 < x) & (x <= 0.3), x <= 0],
                            [lambda x: 1, lambda x: 1 / 0.3 * x, lambda x: 0])

    @staticmethod
    def s3_la(x):
        return np.piecewise(x, [x > -0.4, (-0.8 < x) & (x <= -0.4), x <= -0.8],
                            [lambda x: 0, lambda x: -1 / 0.4 * x - 1, lambda x: 1])

    @staticmethod
    def s4_ne(x):
        return np.piecewise(x, [x > 0, (-0.5 < x) & (x <= 0), x <= -0.5],
                            [lambda x: 0, lambda x: -2 * x, lambda x: 1])

    @staticmethod
    def s4_po(x):
        return np.piecewise(x, [x > 0.5, (0 < x) & (x <= 0.5), x <= 0],
                            [lambda x: 1, lambda x: 2 * x, lambda x: 0])

    @staticmethod
    def s4_sm(x):
        return np.piecewise(x, [x < -0.2, (-0.2 <= x) & (x < 0), (0 <= x) & (x < 0.2), (x >= 0.2)],
                            [lambda x: 0, lambda x: 5 * x + 1, lambda x: -5 * x + 1, lambda x: 0])

    @staticmethod
    def s5_ne(x):
        return np.piecewise(x, [x > 0, (-1.2 < x) & (x <= 0), x <= -1.2],

                            [lambda x: 0, lambda x: -1 / 1.2 * x, lambda x: 1])

    @staticmethod
    def s5_po(x):
        return np.piecewise(x, [x > 1.2, (0 < x) & (x <= 1.2), x <= 0],
                            [lambda x: 1, lambda x: 1 / 1.2 * x, lambda x: 0])
