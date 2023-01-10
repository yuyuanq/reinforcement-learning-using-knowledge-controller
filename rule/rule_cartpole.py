import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MembershipNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 16  #*

        self.fc1 = torch.nn.Linear(1, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, s):
        if len(s.shape) == 1:
            s = s.reshape(-1, 1)

        x = F.leaky_relu(self.fc1(s))
        # x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


def fit_membership_function(model, mf, ob_low, ob_high):
    x = np.linspace(ob_low, ob_high, 1000)
    y = torch.from_numpy(mf(x)).reshape(-1, 1).float()
    x = torch.from_numpy(x).reshape(-1, 1).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss_op = torch.nn.MSELoss()

    for i in range(3000):
        idx = np.random.randint(0, len(x), 32)

        out = model.forward(x[idx])
        loss = loss_op(y[idx], out).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class CartPoleRule:
    def __init__(self):
        self._rule_dict = {0: [[self.s_any, self.s_any, self.s2_ne, self.s3_ne],
                              [self.s_any, self.s_any, self.s2_sm, self.s3_ne],
                              [self.s0_po, self.s1_po, self.s2_sm, self.s3_sm]],
                          1: [[self.s_any, self.s_any, self.s2_po, self.s3_po],
                              [self.s_any, self.s_any, self.s2_sm, self.s3_po],
                              [self.s0_ne, self.s1_ne, self.s2_sm, self.s3_sm]]}
        
        self.rule_dict = {0: nn.ModuleList(), 1: nn.ModuleList()}

        # pre-training
        def plot_mf(model, ob_low, ob_high):
            import matplotlib.pyplot as plt

            x = torch.as_tensor(np.linspace(ob_low, ob_high, 100)).float()
            y = model(x).float()

            plt.figure()
            plt.plot(x.detach().numpy(), y.detach().numpy())
            plt.show()


        ob_high = [2, 2, 2, 2]
        ob_low = [-2, -2, -2, -2]

        for k in self._rule_dict.keys():
            for rule in self._rule_dict[k]:
                tmp = nn.ModuleList()

                for i, mf in enumerate(rule):
                    mf_new = MembershipNetwork()

                    # plot_mf(mf_new, ob_low[i], ob_high[i])
                    fit_membership_function(mf_new, mf, ob_low[i], ob_high[i])
                    tmp.append(mf_new)
                    # plot_mf(mf_new, ob_low[i], ob_high[i])
                    # pass

                self.rule_dict[k].append(tmp)

    @staticmethod
    def s_no(x):
        return np.piecewise(x, [x], [lambda x: 0])

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
