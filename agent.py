import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from controller import Controller
from torch.distributions import Categorical, MultivariateNormal, Normal


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.dense1 = torch.nn.Linear(state_dim, 32)
        self.dense2 = torch.nn.Linear(32, 32)
        self.dense3 = torch.nn.Linear(32, action_dim)

    def forward(self, s, softmax_dim=1):
        x = F.relu(self.dense1(s))
        x = F.relu(self.dense2(x))
        return F.softmax(self.dense3(x), dim=softmax_dim)


class ActorContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_dim, action_scale=1):
        super().__init__()
        self.action_scale = action_scale

        self.fc1 = torch.nn.Linear(state_dim, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc_mu = torch.nn.Linear(32, action_dim)

        torch.nn.init.orthogonal_(self.fc1.weight, 0.1)
        torch.nn.init.orthogonal_(self.fc2.weight, 0.1)
        torch.nn.init.orthogonal_(self.fc_mu.weight, 0.01)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        mu = self.action_scale * torch.tanh(self.fc_mu(x))
        return mu


class Critic(torch.nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.dense1 = torch.nn.Linear(state_dim, 32)
        self.dense2 = torch.nn.Linear(32, 32)
        self.dense3 = torch.nn.Linear(32, 1)

        torch.nn.init.orthogonal_(self.dense1.weight, 0.1)
        torch.nn.init.orthogonal_(self.dense2.weight, 0.1)
        torch.nn.init.orthogonal_(self.dense3.weight, 0.01)

    def forward(self, s):
        x = F.relu(self.dense1(s))
        x = F.relu(self.dense2(x))
        return self.dense3(x)


class PPO(nn.Module):
    def __init__(self, config, state_dim, action_dim):
        super(PPO, self).__init__()
        self.config = config
        self.data = []

        if config.continuous:
            self.action_var = torch.full((action_dim,), self.config.std * self.config.std).cuda()

            if config.no_controller:
                self.actor = ActorContinuous(state_dim, action_dim, action_scale=self.config.action_scale)
            else:
                self.actor = Controller(state_dim, action_dim, config)
                self.p_delta = (self.actor.p_cof - self.actor.p_cof_end) / self.actor.p_total_step
        else:
            if config.no_controller:
                self.actor = Actor(state_dim, action_dim)
            else:
                self.actor = Controller(state_dim, action_dim, config)
                self.p_delta = (self.actor.p_cof - self.actor.p_cof_end) / self.actor.p_total_step

        self.critic = Critic(state_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        self.MseLoss = nn.SmoothL1Loss()

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            if self.config.continuous:
                a_lst.append(a)
            else:
                a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = False if done else True
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.bool), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s_, a_, r_, s_prime_, done_mask_, log_prob_a_ = [x.cuda() for x in self.make_batch()]
        mini_batch = s_.shape[0]  # // 30

        for i in range(self.config.k_epoch):
            with torch.no_grad():
                if self.config.no_gae:
                    rewards = []
                    discounted_reward = 0
                    for reward, is_not_terminal in zip(reversed(r_), reversed(done_mask_)):
                        if not is_not_terminal:
                            discounted_reward = 0
                        discounted_reward = reward + (self.config.gamma * discounted_reward)
                        rewards.insert(0, discounted_reward)

                    rewards = torch.tensor(rewards, dtype=torch.float).cuda()
                    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
                    td_target_ = torch.unsqueeze(rewards, -1)
                    advantage_ = rewards - self.critic(s_)
                else:
                    td_target_ = r_ + self.config.gamma * self.critic(s_prime_) * done_mask_
                    delta = td_target_ - self.critic(s_)
                    delta = delta.cpu().numpy()

                    advantage_lst = []
                    advantage = 0.0
                    for delta_t in delta[::-1]:
                        advantage = self.config.gamma * self.config.lmbda * advantage + delta_t[0]
                        advantage_lst.append([advantage])
                    advantage_lst.reverse()
                    advantage_ = torch.tensor(advantage_lst, dtype=torch.float).cuda()

            for k in range(s_.shape[0] // mini_batch):
                s = s_[mini_batch * k:mini_batch * (k + 1) - 1, :]
                a = a_[mini_batch * k:mini_batch * (k + 1) - 1, :]
                log_prob_a = log_prob_a_[mini_batch * k:mini_batch * (k + 1) - 1, :]
                advantage = advantage_[mini_batch * k:mini_batch * (k + 1) - 1, :]
                td_target = td_target_[mini_batch * k:mini_batch * (k + 1) - 1, :]

                if self.config.continuous:
                    action_mean = self.actor(s)
                    action_var = self.action_var.expand_as(action_mean)
                    cov_mat = torch.diag_embed(action_var).cuda()
                    dist = MultivariateNormal(action_mean, cov_mat)
                    log_pi_a = torch.unsqueeze(dist.log_prob(a), 1)
                    entropy = dist.entropy()
                else:
                    pi = self.actor(s)
                    log_pi_a = torch.log(pi.gather(1, a))
                    entropy = Categorical(pi).entropy()

                ratio = torch.exp(log_pi_a - log_prob_a)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * advantage

                loss = -torch.min(surr1, surr2) + \
                       self.config.mse_cof * self.MseLoss(td_target, self.critic(s)) - \
                       self.config.entropy_cof * entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()

        if not self.config.no_controller:
            self.actor.p_cof = max(self.actor.p_cof - self.p_delta, self.actor.p_cof_end)
