import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # Method 1 (auto)
        # self.rule_dict = nn.ModuleDict({'0': nn.ModuleList([Rule([0, 1, 2, 3], self.config.device)]),
        #                                 '1': nn.ModuleList([Rule([0, 1, 2, 3], self.config.device)])})

        # Method 2 (human designed)
        self.rule_dict = nn.ModuleDict({str(i): nn.ModuleList() for i in range(action_dim)})

        # Define different rules
        if config.env == 'CartPole-v1':
            from rule.rule_cartpole import CartPoleRule as DesignedRule
        elif config.env == 'FlappyBird':
            from rule.rule_flappybird import FlappyBirdRule as DesignedRule
        elif config.env == 'LunarLander-v2':
            from rule.rule_lunarlander import LunarLanderRule as DesignedRule
        elif config.env == 'LunarLanderContinuous-v2':
            from rule.rule_lunarlandercontinuous import LunarLanderContinuousRule as DesignedRule
        elif config.env == 'MountainCarContinuous-v0':
            from rule.rule_mountaincarcontinuous import MountainCarContinuousRule as DesignedRule
        elif config.env == 'MountainCar-v0':
            from rule.rule_mountaincar import MountainCarRule as DesignedRule
        elif config.env == 'Pendulum-v0' or config.env == 'Pendulum-v1':
            from rule.rule_pendulum import PendulumRule as DesignedRule
        else:
            raise ValueError

        self.designed_rule_dic = DesignedRule().rule_dict
        for action_id, rule_list_list in self.designed_rule_dic.items():
            for rule_list in rule_list_list:
                self.rule_dict[str(action_id)].append(
                    HardRuleContinuous(rule_list, self.config.device) if config.continuous else
                    HardRule(rule_list, self.config.device))

        self.hidden_num = 32

        # self.fc1 = nn.Linear(state_dim, self.hidden_num)
        # self.fc2 = nn.Linear(self.hidden_num, self.hidden_num)
        # self.fc3 = nn.Linear(self.hidden_num, action_dim)

        self.w1_fc1 = nn.Linear(state_dim, self.hidden_num)
        self.w1_fc2 = nn.Linear(self.hidden_num, self.hidden_num)
        self.w1_fc3 = nn.Linear(self.hidden_num, self.hidden_num * action_dim)
        self.b1_fc1 = nn.Linear(state_dim, self.hidden_num)
        self.b1_fc2 = nn.Linear(self.hidden_num, self.hidden_num)
        self.b1_fc3 = nn.Linear(self.hidden_num, self.hidden_num)
        self.w2_fc1 = nn.Linear(state_dim, self.hidden_num)
        self.w2_fc2 = nn.Linear(self.hidden_num, self.hidden_num)
        self.w2_fc3 = nn.Linear(self.hidden_num, self.hidden_num * action_dim)
        self.b2_fc1 = nn.Linear(state_dim, self.hidden_num)
        self.b2_fc2 = nn.Linear(self.hidden_num, self.hidden_num)
        self.b2_fc3 = nn.Linear(self.hidden_num, action_dim)

        # torch.nn.init.orthogonal_(self.w1_fc1.weight, 0.1)
        # torch.nn.init.orthogonal_(self.w1_fc2.weight, 0.1)
        # torch.nn.init.orthogonal_(self.w1_fc3.weight, 0.01)
        # torch.nn.init.orthogonal_(self.b1_fc1.weight, 0.1)
        # torch.nn.init.orthogonal_(self.b1_fc2.weight, 0.1)
        # torch.nn.init.orthogonal_(self.b1_fc3.weight, 0.01)
        # torch.nn.init.orthogonal_(self.w2_fc1.weight, 0.1)
        # torch.nn.init.orthogonal_(self.w2_fc2.weight, 0.1)
        # torch.nn.init.orthogonal_(self.w2_fc3.weight, 0.01)
        # torch.nn.init.orthogonal_(self.b2_fc1.weight, 0.1)
        # torch.nn.init.orthogonal_(self.b2_fc2.weight, 0.1)
        # torch.nn.init.orthogonal_(self.b2_fc3.weight, 0.01)

        self.p_cof = config.p_cof
        self.p_cof_end = config.p_cof_end
        self.p_total_step = config.p_total_step

    def forward(self, s):
        # Using rule to get p
        p_list = []
        if self.config.continuous:
            for i in range(self.action_dim):
                wa = [rule(s) for rule in self.rule_dict[str(i)]]
                w = [v[0] for v in wa]
                a = [v[1] for v in wa]
                w = torch.cat(w, 1)

                p_list.append(torch.sum(w * torch.cat(a, 1), 1, keepdim=True))
        else:
            for i in range(self.action_dim):
                rule_list_for_action = [rule(s) for rule in self.rule_dict[str(i)]]
                p_list.append(torch.max(torch.cat(rule_list_for_action, 1), 1, keepdim=True)[0])

        p = torch.cat(p_list, 1)
        # p = torch.ones(s.shape[0], self.action_dim).to(self.config.device)

        # Method 1: Cat network
        # h1 = F.leaky_relu(self.fc1(torch.cat([p.detach(), s], 1)))
        # x = F.leaky_relu(self.fc1(s))
        # x = F.leaky_relu(self.fc2(x))
        # p_out = self.fc3(x)

        # Method 2: Hyper network
        w1 = self.w1_fc3(F.leaky_relu(self.w1_fc2(F.leaky_relu(self.w1_fc1(s))))) \
            .reshape(-1, self.action_dim, self.hidden_num)
        b1 = self.b1_fc3(F.leaky_relu(self.b1_fc2(F.leaky_relu(self.b1_fc1(s))))) \
            .reshape(-1, 1, self.hidden_num)
        w2 = self.w2_fc3(F.leaky_relu(self.w2_fc2(F.leaky_relu(self.w2_fc1(s))))) \
            .reshape(-1, self.hidden_num, self.action_dim)
        b2 = self.b2_fc3(F.leaky_relu(self.b2_fc2(F.leaky_relu(self.b2_fc1(s))))) \
            .reshape(-1, 1, self.action_dim)
        p_out = (torch.bmm(F.leaky_relu(
            torch.bmm(p.detach().reshape(-1, 1, self.action_dim), w1) + b1), w2) + b2).reshape(-1, self.action_dim)

        if self.config.continuous:
            p_prime = self.config.action_scale * torch.tanh(p_out)
            return p * self.p_cof + p_prime * (1 - self.p_cof)
        else:
            p_prime = torch.sigmoid(p_out)
            return F.softmax((p * self.p_cof + p_prime * (1 - self.p_cof)) / (1 / 10))


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
        self.dense1 = torch.nn.Linear(1, 32)
        self.dense2 = torch.nn.Linear(32, 32)
        self.dense3 = torch.nn.Linear(32, 1)

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
        return torch.min(membership_all, dim=1, keepdim=True)[0]
