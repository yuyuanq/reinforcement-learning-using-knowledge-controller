import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from controller import Controller


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.dense1 = torch.nn.Linear(state_dim, 32)
        self.dense2 = torch.nn.Linear(32, 32)
        self.dense3 = torch.nn.Linear(32, action_dim)

    def forward(self, s, softmax_dim=1):
        x = F.leaky_relu(self.dense1(s))
        x = F.leaky_relu(self.dense2(x))
        return F.softmax(self.dense3(x), dim=softmax_dim)


class Critic(torch.nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.dense1 = torch.nn.Linear(state_dim, 32)
        self.dense2 = torch.nn.Linear(32, 32)
        self.dense3 = torch.nn.Linear(32, 1)

    def forward(self, s):
        x = F.leaky_relu(self.dense1(s))
        x = F.leaky_relu(self.dense2(x))
        return self.dense3(x)


class PPO(nn.Module):
    def __init__(self, config, state_dim, action_dim):
        super(PPO, self).__init__()
        self.config = config
        self.data = []

        if config.no_controller:
            self.actor = Actor(state_dim, action_dim)
        else:
            self.actor = Controller(state_dim, action_dim)
            self.p_delta = (self.actor.p_cof - self.actor.p_cof_end) / self.actor.p_total_step
        self.critic = Critic(state_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = [x.cuda() for x in self.make_batch()]

        for i in range(self.config.k_epoch):
            td_target = r + self.config.gamma * self.critic(s_prime) * done_mask
            delta = td_target - self.critic(s)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.config.gamma * self.config.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).cuda()

            # pi = self.actor(s, softmax_dim=1) if self.config.no_controller else self.actor(s)
            pi = self.actor(s)

            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        if not self.config.no_controller:
            self.actor.p_cof = max(self.actor.p_cof - self.p_delta, self.actor.p_cof_end)
