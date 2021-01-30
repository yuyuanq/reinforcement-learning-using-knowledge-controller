import matplotlib.pyplot as plt
import numpy as np


def check(func, range=10):
    x = np.linspace(-range, range, 1000)
    plt.plot(x, list(map(func, x)))
    plt.show()


class FlappyBirdRule:
    def __init__(self):
        self.rule_dict = {
            0: [[self.s0_s, self.s_any, self.s_any, self.s_any, self.s_any],
                [self.s_any, self.s1_ne, self.s_any, self.s3_ne, self.s_any]],
            1: [[self.s0_l, self.s1_po, self.s_any, self.s_any, self.s_any],
                [self.s_any, self.s1_po, self.s_any, self.s_any, self.s4_po]]}

    @staticmethod
    def s_any(x):
        return np.piecewise(x, [x], [lambda x: 1])

    @staticmethod
    def s0_l(x):
        return np.piecewise(x, [x > 2.5, (2 < x) & (x <= 2.5), x <= 2],
                            [lambda x: 1, lambda x: (1 / 0.5) * x - 4, lambda x: 0])

    @staticmethod
    def s0_s(x):
        return np.piecewise(x, [x > 2, (1.5 < x) & (x <= 1.5), x <= 1.5],
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


class CartPoleRule:
    def __init__(self):
        self.rule_dict = {0: [[self.s_any, self.s_any, self.s2_ne, self.s3_ne],
                              [self.s_any, self.s_any, self.s2_sm, self.s3_ne],
                              [self.s0_po, self.s1_po, self.s2_sm, self.s3_sm]],
                          1: [[self.s_any, self.s_any, self.s2_po, self.s3_po],
                              [self.s_any, self.s_any, self.s2_sm, self.s3_po],
                              [self.s0_ne, self.s1_ne, self.s2_sm, self.s3_sm]]}

    @staticmethod
    def s_any(x):
        return np.piecewise(x, [x], [lambda x: 1])

    @staticmethod
    def s2_ne(x):
        return np.piecewise(x, [x < -1, (-1 <= x) & (x < 0), x >= 0],
                            [lambda x: 1, lambda x: -x, lambda x: 0])

    @staticmethod
    def s2_sm(x):
        return np.piecewise(x, [x < -1, (-1 <= x) & (x < 0), (0 <= x) & (x < 1), x >= 1],
                            [lambda x: 0, lambda x: x + 1, lambda x: -x + 1, lambda x: 0])

    @staticmethod
    def s2_po(x):
        return np.piecewise(x, [x < 0, (0 <= x) & (x < 1), x >= 1],
                            [lambda x: 0, lambda x: x, lambda x: 1])

    @staticmethod
    def s3_ne(x):
        return np.piecewise(x, [x < -1, (-1 <= x) & (x < 0), x >= 0],
                            [lambda x: 1, lambda x: -x, lambda x: 0])

    @staticmethod
    def s3_sm(x):
        return np.piecewise(x, [x < -0.5, (-0.5 <= x) & (x < 0), (0 <= x) & (x < 0.5), x >= 0.5],
                            [lambda x: 0, lambda x: 2 * x + 1, lambda x: -2 * x + 1, lambda x: 0])

    @staticmethod
    def s3_po(x):
        return np.piecewise(x, [x < 0, (0 <= x) & (x < 1), x >= 1],
                            [lambda x: 0, lambda x: x, lambda x: 1])

    @staticmethod
    def s0_ne(x):
        return np.piecewise(x, [x < -2, (-2 <= x) & (x < 0), x >= 0],
                            [lambda x: 1, lambda x: -0.5 * x, lambda x: 0])

    @staticmethod
    def s0_po(x):
        return np.piecewise(x, [x < 0, (0 <= x) & (x < 2), x >= 2],
                            [lambda x: 0, lambda x: 0.5 * x, lambda x: 1])

    @staticmethod
    def s1_ne(x):
        return np.piecewise(x, [x < -1, (-1 <= x) & (x < 0), x >= 0],
                            [lambda x: 1, lambda x: -x, lambda x: 0])

    @staticmethod
    def s1_po(x):
        return np.piecewise(x, [x < 0, (0 <= x) & (x < 1), x >= 1],
                            [lambda x: 0, lambda x: x, lambda x: 1])


if __name__ == '__main__':
    rule = FlappyBirdRule()
    check(rule.s4_po, range=2)
