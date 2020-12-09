import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Controller(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.rule_dict = nn.ModuleDict({'0': nn.ModuleList([Rule('!', [0, 1, 2, 3]),
                                                            Rule('!', [0, 1, 2, 3])]),
                                        '1': nn.ModuleList([Rule('!', [0, 1, 2, 3]),
                                                            Rule('!', [0, 1, 2, 3])])})

    def forward(self, s):  # get action distribution
        strength_all = torch.zeros((s.shape[0], self.action_dim)).cuda()

        for i in range(self.action_dim):
            rule_list_for_action = [rule(s).reshape(-1, 1) for rule in self.rule_dict[str(i)]]
            strength_all[:, i] = torch.max(torch.cat(rule_list_for_action, 1), 1)[0]  # max
        return F.softmax(strength_all, dim=1)


class Rule(torch.nn.Module):
    def __init__(self, id, state_id):
        super().__init__()
        self.id = id
        self.state_id = state_id

        self.membership_network_list = nn.ModuleList()
        for i in range(len(state_id)):
            self.membership_network_list.append(MembershipNetwork())

    def forward(self, s):  # get strength
        membership_all = torch.zeros((s.shape[0], len(self.state_id))).cuda()

        for i in range(len(self.state_id)):
            mf = self.membership_network_list[i]
            membership_all[:, i] = torch.squeeze(mf(s[:, self.state_id[i]].reshape(-1, 1)))
        return torch.min(membership_all, dim=1)[0]  # min


class MembershipNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(1, 8)
        self.dense2 = torch.nn.Linear(8, 8)
        self.dense3 = torch.nn.Linear(8, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def forward(self, s):  # get membership
        x = F.relu(self.dense1(s))
        x = F.relu(self.dense2(x))
        return torch.sigmoid(self.dense3(x))
