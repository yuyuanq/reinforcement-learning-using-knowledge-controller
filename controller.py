import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # # Method 1 (auto)  For mc
        # self.rule_dict = nn.ModuleDict({'0': nn.ModuleList([Rule([0, 1], self.config.device)]),
        #                                 '1': nn.ModuleList([Rule([0, 1], self.config.device)]),
        #                                 '2': nn.ModuleList([Rule([0, 1], self.config.device)])})

        # Method 1 (auto)  For cp
        # self.rule_dict = nn.ModuleDict({'0': nn.ModuleList([Rule([0, 1, 2, 3], self.config.device),
        #                                                     Rule([0, 1, 2, 3], self.config.device)]),
        #                                 '1': nn.ModuleList([Rule([0, 1, 2, 3], self.config.device),
        #                                                     Rule([0, 1, 2, 3], self.config.device)])})

        self.rule_dict = nn.ModuleDict({'0': nn.ModuleList([Rule([0, 1, 2, 3], self.config.device)]),
                                        '1': nn.ModuleList([Rule([0, 1, 2, 3], self.config.device)])})


        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, s):
        strength_all = torch.zeros((s.shape[0], self.action_dim)).to(self.config.device)

        for i in range(self.action_dim):
            rule_list_for_action = [rule(s).reshape(-1, 1) for rule in self.rule_dict[str(i)]]
            strength_all[:, i] = torch.max(torch.cat(rule_list_for_action, 1), 1)[0]  # max
        return F.softmax(strength_all * 5, dim=1)


class HardRule(torch.nn.Module):
    def __init__(self, rule_list, device):
        super().__init__()
        self.rule_list = rule_list
        self.w_list = nn.Parameter(torch.ones((len(rule_list) + 1)))
        self.device = device

    def forward(self, s):
        return self.w_list[-1] * torch.min(torch.cat(
            [self.w_list[i] * torch.unsqueeze(torch.as_tensor(self.rule_list[i](s[:, i].cpu().numpy())), 1).to(self.device)
             for i in range(len(self.rule_list))], 1), 1, keepdim=True)[0]


class HardRuleContinuous(torch.nn.Module):
    def __init__(self, rule_list, device):
        super().__init__()
        self.rule_list = rule_list
        self.w_list = nn.Parameter(torch.ones((len(rule_list))))
        self.device = device

    def forward(self, s):
        w = self.w_list[-1] * torch.min(torch.cat(
            [self.w_list[i] * torch.unsqueeze(torch.as_tensor(self.rule_list[i](s[:, i].cpu().numpy())), 1).to(self.device)
             for i in range(len(self.rule_list) - 1)], 1), 1, keepdim=True)[0]
        a = self.rule_list[-1](w)
        return w, a


class MembershipNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(1, 16)
        self.dense2 = torch.nn.Linear(16, 16)
        self.dense3 = torch.nn.Linear(16, 1)

    def forward(self, s):
        x = F.leaky_relu(self.dense1(s))
        x = F.leaky_relu(self.dense2(x))
        return torch.sigmoid(self.dense3(x))


class Rule(torch.nn.Module):
    def __init__(self, state_id_list, device):
        super().__init__()
        self.state_id = state_id_list
        self.device = device

        self.membership_network_list = nn.ModuleList()
        for i in range(len(state_id_list)):
            self.membership_network_list.append(MembershipNetwork())

    def forward(self, s):
        membership_all = torch.zeros((s.shape[0], len(self.state_id))).to(self.device)

        for i in range(len(self.state_id)):
            mf = self.membership_network_list[i]
            membership_all[:, i] = torch.squeeze(mf(s[:, self.state_id[i]].reshape(-1, 1)))

        min_strength = torch.min(membership_all, dim=1, keepdim=True)[0]
        return min_strength  # + torch.prod(membership_all, dim=1, keepdim=True) / min_strength
