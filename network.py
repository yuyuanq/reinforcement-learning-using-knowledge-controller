import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.dense1 = torch.nn.Linear(state_dim, 32)
        self.dense2 = torch.nn.Linear(32, 32)
        self.dense3 = torch.nn.Linear(32, action_dim)

        nn.init.orthogonal_(self.dense1.weight, 0.1)
        nn.init.orthogonal_(self.dense2.weight, 0.1)
        nn.init.orthogonal_(self.dense3.weight, 0.01)

    def forward(self, s, softmax_dim=0):
        x = F.leaky_relu(self.dense1(s))
        x = F.leaky_relu(self.dense2(x))
        return F.softmax(self.dense3(x), dim=softmax_dim)


class Critic(torch.nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.dense1 = torch.nn.Linear(state_dim, 32)
        self.dense2 = torch.nn.Linear(32, 32)
        self.dense3 = torch.nn.Linear(32, 1)

        nn.init.orthogonal_(self.dense1.weight, 0.1)
        nn.init.orthogonal_(self.dense2.weight, 0.1)
        nn.init.orthogonal_(self.dense3.weight, 0.01)

    def forward(self, s):
        x = F.leaky_relu(self.dense1(s))
        x = F.leaky_relu(self.dense2(x))
        return self.dense3(x)


class MembershipNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(1, 32)
        self.dense2 = torch.nn.Linear(32, 32)
        self.dense3 = torch.nn.Linear(32, 1)

        nn.init.orthogonal_(self.dense1.weight, 0.1)
        nn.init.orthogonal_(self.dense2.weight, 0.1)
        nn.init.orthogonal_(self.dense3.weight, 0.01)

    def forward(self, s):
        x = F.leaky_relu(self.dense1(s))
        x = F.leaky_relu(self.dense2(x))
        return torch.sigmoid(self.dense3(x))
