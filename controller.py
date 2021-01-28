import torch
import torch.nn as nn
import torch.nn.functional as F
from designed_rule import CartPoleRule


class Controller(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim

        # Method 1 (auto)
        # self.rule_dict = nn.ModuleDict({'0': nn.ModuleList([Rule([0, 1, 2, 3])]),
        #                                 '1': nn.ModuleList([Rule([0, 1, 2, 3])])})

        # Method 2 (human designed)
        self.rule_dict = nn.ModuleDict({'0': nn.ModuleList(), '1': nn.ModuleList()})
        self.designed_rule_dic = CartPoleRule().rule_dict
        for action_id, rule_list_list in self.designed_rule_dic.items():
            for rule_list in rule_list_list:
                self.rule_dict[str(action_id)].append(HardRule(rule_list))

        self.hidden_num = 32

        # Cat network
        # self.fc1 = nn.Linear(action_dim + state_dim, self.hidden_num)
        # self.fc2 = nn.Linear(self.hidden_num, self.hidden_num)
        # self.fc3 = nn.Linear(self.hidden_num, action_dim)

        # Hyper network
        self.w1_fc1 = nn.Linear(state_dim, self.hidden_num)
        self.w1_fc2 = nn.Linear(self.hidden_num, self.hidden_num)
        self.w1_fc3 = nn.Linear(self.hidden_num, self.hidden_num * action_dim)

        self.b1_fc1 = nn.Linear(state_dim, self.hidden_num)
        self.b1_fc2 = nn.Linear(self.hidden_num, self.hidden_num)
        self.b1_fc3 = nn.Linear(self.hidden_num, self.hidden_num)

        self.w2_fc1 = nn.Linear(state_dim, self.hidden_num)
        self.w2_fc2 = nn.Linear(self.hidden_num, self.hidden_num)
        self.w2_fc3 = nn.Linear(self.hidden_num, self.hidden_num * action_dim)

        self.p_cof = 0.7
        self.p_cof_end = 0.1
        self.p_total_step = 2000

    def forward(self, s):
        p_list = []
        for i in range(self.action_dim):
            rule_list_for_action = [rule(s) for rule in self.rule_dict[str(i)]]
            p_list.append(torch.max(torch.cat(rule_list_for_action, 1), 1, keepdim=True)[0])

        p = torch.cat(p_list, 1)

        # p_prime = F.leaky_relu(self.fc1(torch.cat([p.detach(), s], 1)))
        # p_prime = F.leaky_relu(self.fc2(p_prime))
        # p_prime = torch.sigmoid(self.fc3(p_prime))

        w1 = self.w1_fc3(F.leaky_relu(self.w1_fc2(F.leaky_relu(self.w1_fc1(s))))) \
            .reshape(-1, self.action_dim, self.hidden_num)
        b1 = self.b1_fc3(F.leaky_relu(self.b1_fc2(F.leaky_relu(self.b1_fc1(s))))) \
            .reshape(-1, 1, self.hidden_num)
        w2 = self.w2_fc3(F.leaky_relu(self.w2_fc2(F.leaky_relu(self.w2_fc1(s))))) \
            .reshape(-1, self.hidden_num, self.action_dim)

        p_prime = torch.sigmoid(
            torch.bmm(F.leaky_relu(torch.bmm(p.detach().reshape(-1, 1, self.action_dim), w1) + b1), w2)).\
            reshape(-1, self.action_dim)

        return torch.squeeze(F.softmax((p * self.p_cof + p_prime * (1 - self.p_cof)) / (1 / 10), dim=1))


class HardRule(torch.nn.Module):
    def __init__(self, rule_list):
        super().__init__()
        self.rule_list = rule_list
        self.w_list = nn.Parameter(torch.ones((len(rule_list) + 1)))

    def forward(self, s):
        return self.w_list[-1] * torch.min(torch.cat(
            [self.w_list[i] * torch.unsqueeze(torch.as_tensor(self.rule_list[i](s[:, i].cpu().numpy())), 1).cuda()
             for i in range(len(self.rule_list))], 1), 1, keepdim=True)[0]


class MembershipNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(1, 32)
        self.dense2 = torch.nn.Linear(32, 32)
        self.dense3 = torch.nn.Linear(32, 1)

    def forward(self, s):
        x = F.leaky_relu(self.dense1(s))
        x = F.leaky_relu(self.dense2(x))
        return torch.sigmoid(self.dense3(x))


class Rule(torch.nn.Module):
    def __init__(self, state_id_list):
        super().__init__()
        self.state_id = state_id_list

        self.membership_network_list = nn.ModuleList()
        for i in range(len(state_id_list)):
            self.membership_network_list.append(MembershipNetwork())

    def forward(self, s):
        membership_all = torch.zeros((s.shape[0], len(self.state_id))).cuda()

        for i in range(len(self.state_id)):
            mf = self.membership_network_list[i]
            membership_all[:, i] = torch.squeeze(mf(s[:, self.state_id[i]].reshape(-1, 1)))
        return torch.min(membership_all, dim=1, keepdim=True)[0]
