import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Controller(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # key is the id of action
        self.rule_dict = {0: [Rule('a0_0', state_dim, [0, 1, 2, 3]).cuda(), Rule('a0_1', state_dim, [0, 1, 2, 3]).cuda()],
                          1: [Rule('a1_0', state_dim, [0, 1, 2, 3]).cuda(), Rule('a1_1', state_dim, [0, 1, 2, 3]).cuda()]}
        assert max(self.rule_dict.keys()) == action_dim - 1

    def forward(self, s):
        strength_all = torch.zeros((s.shape[0], self.action_dim)).cuda()

        for i in range(self.action_dim):
            strength_all[:, i] = torch.mean(torch.cat([rule(s).reshape(-1, 1) for rule in self.rule_dict[i]], 1), 1)[0]  # max
        return F.softmax(strength_all, dim=1)  # output prob of controller


class Rule(torch.nn.Module):
    def __init__(self, id, state_dim, state_id):
        super().__init__()
        self.id = id
        self.state_dim = state_dim
        self.state_id = state_id

        self.membership_network_list = []
        for i in range(len(state_id)):
            self.membership_network_list.append(MembershipNetwork().cuda())

    def forward(self, s):  # get strength
        membership_all = torch.zeros((s.shape[0], len(self.state_id))).cuda()

        for i in range(len(self.state_id)):
            mf = self.membership_network_list[i]
            membership_all[:, i] = torch.squeeze(mf(s[:, self.state_id[i]].reshape(-1, 1)))
        return torch.mean(membership_all, dim=1)[0]  # min


class MembershipNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(1, 32)
        self.dense2 = torch.nn.Linear(32, 32)
        self.dense3 = torch.nn.Linear(32, 1)

        # he initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def forward(self, s):  # get membership
        x = F.relu(self.dense1(s))
        x = F.relu(self.dense2(x))
        return F.sigmoid(self.dense3(x))
