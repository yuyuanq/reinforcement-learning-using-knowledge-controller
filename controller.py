import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(torch.nn.Module):

    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # For cp
        # self.rule_dict = nn.ModuleDict({'0': nn.ModuleList([Rule([0, 1, 2, 3], self.config.device),
        #                                                     Rule([0, 1, 2, 3], self.config.device)]),
        #                                 '1': nn.ModuleList([Rule([0, 1, 2, 3], self.config.device),
        #                                                     Rule([0, 1, 2, 3], self.config.device)])})

        self.rule_dict = nn.ModuleDict({
            '0':
            nn.ModuleList([Rule([0, 1, 2, 3], self.config.device)]),
            '1':
            nn.ModuleList([Rule([0, 1, 2, 3], self.config.device)])
        })

        # for fb
        # self.rule_dict = nn.ModuleDict({
        #     '0':
        #     nn.ModuleList([Rule([0, 1, 2, 3, 4], self.config.device)]),
        #     '1':
        #     nn.ModuleList([Rule([0, 1, 2, 3, 4], self.config.device)])
        # })

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, s):
        strength_all = torch.zeros(
            (s.shape[0], self.action_dim)).to(self.config.device)

        for i in range(self.action_dim):
            rule_list_for_action = [
                rule(s).reshape(-1, 1) for rule in self.rule_dict[str(i)]
            ]
            strength_all[:, i] = torch.max(torch.cat(rule_list_for_action, 1),
                                           1)[0]  # max
        return F.softmax(strength_all * 5, dim=1)


class MembershipNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, s):
        x = F.leaky_relu(self.fc1(s))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


class Rule(torch.nn.Module):

    def __init__(self, state_id_list, device):
        super().__init__()
        self.state_id = state_id_list
        self.device = device

        self.membership_network_list = nn.ModuleList()
        for _ in range(len(state_id_list)):
            self.membership_network_list.append(MembershipNetwork())

    def forward(self, s):
        membership_all = torch.zeros(
            (s.shape[0], len(self.state_id))).to(self.device)

        for i in range(len(self.state_id)):
            mf = self.membership_network_list[i]
            membership_all[:, i] = torch.squeeze(
                mf(s[:, self.state_id[i]].reshape(-1, 1)))

        # Method 1: use soft min
        weight = torch.zeros_like(membership_all)
        for i in range(weight.shape[1]):
            weight[:, i] = 1 / membership_all[:, i]
        
        min_strength = torch.zeros((s.shape[0], 1))
        for i in range(membership_all.shape[1]):
            min_strength += torch.unsqueeze(membership_all[:, i] * weight[:, i], 1)
        
        min_strength /= torch.sum(weight, dim=1, keepdim=True)
        min_strength[torch.isnan(min_strength)] = 1e-20

        # Method 2: use min
        # min_strength = torch.min(membership_all, dim=1, keepdim=True)[0]

        return min_strength
