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
            rule_list_for_action = [
                rule(s).reshape(-1, 1) for rule in self.rule_dict[str(i)]
            ]
            strength_all[:, i] = torch.max(
                torch.cat(rule_list_for_action, 1), 1)[0]  # max
        return F.softmax(strength_all * 10, dim=1)


class MembershipNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, s):
        x = F.leaky_relu(self.fc1(s))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


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


class Rule(torch.nn.Module):

    def __init__(self, state_id_list, device):
        super().__init__()
        self.state_id = state_id_list
        self.device = device

        self.membership_network_list = nn.ModuleList()
        for _ in range(len(state_id_list)):
            self.membership_network_list.append(MembershipNetwork())

        p_select_tensor = torch.ones((len(state_id_list), 2)) * 10  # TODO: adjust
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
            # soft_weight[:, i] = -(gumbel_select[i, 1] * membership_all[:, i])  # TODO: adjust
            soft_weight[:, i] = gumbel_select[i, 1] / membership_all[:, i]

        soft_weight = softmax(soft_weight * 0.1, gumbel_select)  # TODO: adjust
        # soft_weight = torch.softmax(soft_weight, 1)

        min_strength = torch.zeros((s.shape[0], 1))
        for i in range(membership_all.shape[1]):
            min_strength += torch.unsqueeze(
                membership_all[:, i] * soft_weight[:, i] * gumbel_select[i, 1], 1)
        # print(self.p_select, gumbel_select, soft_weight, membership_all, min_strength)

        if torch.any(torch.isnan(min_strength)):
            self.debug_count_1 += 1
            print('*'*30 + 'Warning: NaN is not a valid, count_1: {}'.format(self.debug_count_1))
            print(self.p_select, gumbel_select, soft_weight, membership_all, min_strength)
            min_strength = torch.where(torch.isnan(min_strength), torch.full_like(min_strength, 0), min_strength)

        # Method 2: use min
        # min_strength = torch.min(membership_all, dim=1, keepdim=True)[0]

        return min_strength
