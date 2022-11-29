import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


SMALL_NUM = 0.015


class Controller(torch.nn.Module):

    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # For cp
        # self.rule_dict = nn.ModuleDict({
        #     '0':
        #     nn.ModuleList([Rule([0, 1, 2, 3], self.config.device)]),
        #     '1':
        #     nn.ModuleList([Rule([0, 1, 2, 3], self.config.device)])
        # })

        # for fb
        # self.rule_dict = nn.ModuleDict({
        #     '0':
        #     nn.ModuleList([Rule([0, 1, 2, 3, 4], self.config.device)]),
        #     '1':
        #     nn.ModuleList([Rule([0, 1, 2, 3, 4], self.config.device)])
        # })

        # for ll
        # self.rule_dict = nn.ModuleDict({
        #     '0':
        #     nn.ModuleList([Rule([0, 1, 2, 3, 4, 5, 6, 7], self.config.device)]),
        #     '1':
        #     nn.ModuleList([Rule([0, 1, 2, 3, 4, 5, 6, 7], self.config.device)]),
        #     '2':
        #     nn.ModuleList([Rule([0, 1, 2, 3, 4, 5, 6, 7], self.config.device)]),
        #     '3':
        #     nn.ModuleList([Rule([0, 1, 2, 3, 4, 5, 6, 7], self.config.device)])
        # })

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, s):
        strength_all = torch.zeros(
            (s.shape[0], self.action_dim)).to(self.config.device)

        for i in range(self.action_dim):
            rule_list_for_action = [rule(s).reshape(-1, 1) for rule in self.rule_dict[str(i)]]
            strength_all[:, i] = torch.max(torch.cat(rule_list_for_action, 1), 1)[0]

        return F.softmax(strength_all * 10, dim=1)


class FuzzyTreeController(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # For cp
        tree_height = 2
        leaves_num = 2 ** tree_height

        self.rule_tree = nn.ModuleList([Rule([0, 1, 2, 3], self.config.device) for _ in range(int(2 ** tree_height - 1))])
        self.leaves_params = nn.Parameter(
            torch.randn((leaves_num, action_dim)))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, s):
        rule_tree_strength = [self.rule_tree[i](s) for i in range(len(self.rule_tree))]
        leaves_distribution = torch.softmax(self.leaves_params, 1)

        # output = 0
        # output += torch.mm(rule_tree_strength[0] * rule_tree_strength[1], torch.unsqueeze(leaves_distribution[0, :], 0))
        # output += torch.mm(rule_tree_strength[0] * (1-rule_tree_strength[1]), torch.unsqueeze(leaves_distribution[1, :], 0))
        # output += torch.mm((1-rule_tree_strength[0]) * rule_tree_strength[2], torch.unsqueeze(leaves_distribution[2, :], 0))
        # output += torch.mm((1-rule_tree_strength[0]) * (1-rule_tree_strength[2]), torch.unsqueeze(leaves_distribution[3, :], 0))

        path_p = torch.cat([rule_tree_strength[0] * rule_tree_strength[1],
                  rule_tree_strength[0] * (1-rule_tree_strength[1]),
                  (1-rule_tree_strength[0]) * rule_tree_strength[2],
                  (1-rule_tree_strength[0]) * (1-rule_tree_strength[2])], axis=1)
        idx = torch.argmax(path_p, axis=1)

        output = leaves_distribution[idx, :]
        return output


class MembershipNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 16  #*

        self.fc1 = torch.nn.Linear(1, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 1)

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

        p_select_tensor = torch.ones((len(state_id_list), 2)) * 10  # * adjust
        p_select_tensor[:, 0] = 0

        self.p_select = nn.Parameter(p_select_tensor)
        self.debug_count_1 = 0

    def forward(self, s):
        membership_all = torch.zeros(
            (s.shape[0], len(self.state_id))).to(self.device)

        for i in range(len(self.state_id)):
            mf = self.membership_network_list[i]
            membership_all[:, i] = torch.squeeze(
                mf(s[:, self.state_id[i]].reshape(-1, 1)))

        # Method 1: use soft min
        membership_all = torch.where(
            membership_all < SMALL_NUM, torch.as_tensor(SMALL_NUM), membership_all)
        soft_weight = torch.zeros_like(membership_all)
        gumbel_select = gumbel_softmax(self.p_select, hard=True)

        for i in range(soft_weight.shape[1]):
            # soft_weight[:, i] = -(gumbel_select[i, 1] * membership_all[:, i])
            soft_weight[:, i] = gumbel_select[i, 1] / membership_all[:, i]

        soft_weight = softmax(soft_weight * 0.1, gumbel_select)
        # soft_weight = torch.softmax(soft_weight, 1)

        strength = torch.zeros((s.shape[0], 1))
        for i in range(membership_all.shape[1]):
            strength += torch.unsqueeze(
                membership_all[:, i] * soft_weight[:, i] * gumbel_select[i, 1], 1)
        # print(self.p_select, gumbel_select, soft_weight, membership_all, min_strength)

        if torch.any(torch.isnan(strength)):
            self.debug_count_1 += 1
            print(
                '*'*30 + 'Warning: NaN is not a valid, count_1: {}'.format(self.debug_count_1))
            print(self.p_select, gumbel_select,
                  soft_weight, membership_all, strength)
            strength = torch.where(torch.isnan(
                strength), torch.full_like(strength, 0), strength)

        # Method 2: use min
        # strength = torch.min(membership_all, dim=1, keepdim=True)[0]

        # Method 3: use mul
        # strength = torch.prod(membership_all, dim=1, keepdim=True)

        return strength


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # print(logits[0],"a")
    # print(len(argmax_acs),argmax_acs[0])
    if eps == 0.0:
        return argmax_acs

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb


def gumbel_softmax(logits, temperature=1, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        #print(y_hard[0], "random")
        y = (y_hard - y).detach() + y
    return y


def softmax(X, gumbel_select):
    X_exp = X.exp()
    if sum(gumbel_select[:, 1]) > SMALL_NUM:
        X_exp = X_exp * gumbel_select[:, 1]

    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition
