import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from controller import Controller, FuzzyTreeController
from torch.distributions import Categorical, MultivariateNormal, Normal
import SDT


HIDDEN_SIZE = 128  #*

class Actor(torch.nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, HIDDEN_SIZE)
        # self.fc2 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = torch.nn.Linear(HIDDEN_SIZE, action_dim)

    def forward(self, s, softmax_dim=1):
        x = F.relu(self.fc1(s))
        # x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=softmax_dim)


class ActorMixed(torch.nn.Module):

    def __init__(self, state_dim, action_dim, source_model):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, HIDDEN_SIZE)
        # self.fc2 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = torch.nn.Linear(HIDDEN_SIZE, action_dim)

        self.source_model = source_model
        self.source_actor = lambda x: self.source_model.forward(x, LogProb=False)[1]

        self.w = 0.9
        self.w_target = 0.1
        self.w_epi = 3000

        self.w_init = self.w

    def forward(self, s, ep_count=None, softmax_dim=1):
        if ep_count is not None:
            self.w = max(self.w_init - ep_count * (self.w_init - self.w_target) / self.w_epi, self.w_target)

        x = F.relu(self.fc1(s))
        # x = F.relu(self.fc2(x))
        target_model_output = F.softmax(self.fc3(x), dim=softmax_dim)
        
        c1 = self.w * self.source_actor(s).detach()
        c2 = (1 - self.w) * target_model_output

        if ep_count is not None and ep_count % 100 == 0:
            print(self.w)
            pass

        return c1 + c2
        # return target_model_output


class ActorContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_dim, action_scale=1):
        super().__init__()
        self.action_scale = action_scale

        self.fc1 = torch.nn.Linear(state_dim, HIDDEN_SIZE)
        self.fc2 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc_mu = torch.nn.Linear(HIDDEN_SIZE, action_dim)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        mu = self.action_scale * torch.tanh(self.fc_mu(x))
        return mu


class Critic(torch.nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, HIDDEN_SIZE)
        # self.fc2 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc_v = torch.nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        # x = F.relu(self.fc2(x))
        return self.fc_v(x)


class PPO(nn.Module):
    def __init__(self, config, state_dim, action_dim, source_filepath):
        super(PPO, self).__init__()
        self.config = config
        self.data = []

        if config.continuous:
            self.action_var = torch.full(
                (action_dim, ),
                self.config.std * self.config.std).to(config.device)

            if config.no_controller:
                self.actor = ActorContinuous(
                    state_dim,
                    action_dim,
                    action_scale=self.config.action_scale)
            else:
                self.actor = Controller(state_dim, action_dim, config)
        else:
            if config.no_controller:
                self.actor = Actor(state_dim, action_dim)
            else:
                self.policy = SDT.SDT()
                self.actor = lambda x: self.policy.forward(x, LogProb=False)[1]

        self.critic = Critic(state_dim)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        
        def helper(string):
            if string.startswith('policy.'):
                return string[7:]
            else:
                return string

        if source_filepath is not None:
            from main import model_discretization
            # only load actor params
            pretrained_dict = torch.load(source_filepath)
            net2_dict = self.policy.state_dict()
            pretrained_dict = {helper(k): v for k, v in pretrained_dict.items() if helper(k) in net2_dict.keys()}
            net2_dict.update(pretrained_dict)
            self.policy.load_state_dict(net2_dict)
            model_discretization(self.policy)

            self.actor = ActorMixed(state_dim, action_dim, self.policy)

        self.optimizer = optim.Adam(list(self.parameters()), lr=self.config.learning_rate)
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
        s_, a_, r_, s_prime_, done_mask_, log_prob_a_ = [
            x.to(self.config.device) for x in self.make_batch()
        ]
        # print(s_.shape)
        mini_batch = s_.shape[0] // (1 if not self.config.use_minibatch else self.config.minibatch)

        for i in range(self.config.k_epoch):
            with torch.no_grad():
                if self.config.no_gae:
                    rewards = []
                    discounted_reward = 0
                    for reward, is_not_terminal in zip(reversed(r_),
                                                       reversed(done_mask_)):
                        if not is_not_terminal:
                            discounted_reward = 0
                        discounted_reward = reward + (self.config.gamma *
                                                      discounted_reward)
                        rewards.insert(0, discounted_reward)

                    rewards = torch.tensor(rewards, dtype=torch.float).to(
                        self.config.device)
                    rewards = (rewards - rewards.mean()) / (rewards.std() +
                                                            1e-5)
                    td_target_ = torch.unsqueeze(rewards, -1)
                    advantage_ = torch.unsqueeze(rewards, -1) - self.critic(s_)
                else:
                    td_target_ = r_ + self.config.gamma * self.critic(
                        s_prime_) * done_mask_
                    delta = td_target_ - self.critic(s_)
                    delta = delta.cpu().numpy()

                    advantage_lst = []
                    advantage = 0.0
                    for delta_t in delta[::-1]:
                        advantage = self.config.gamma * self.config.lmbda * advantage + delta_t[
                            0]
                        advantage_lst.append([advantage])
                    advantage_lst.reverse()
                    advantage_ = torch.tensor(advantage_lst, dtype=torch.float).to(self.config.device)

            # In most cases, do not use mini batch
            for k in range(s_.shape[0] // mini_batch):
                s = s_[mini_batch * k:mini_batch * (k + 1), :]
                a = a_[mini_batch * k:mini_batch * (k + 1), :]
                log_prob_a = log_prob_a_[mini_batch * k:mini_batch *
                                         (k + 1), :]
                advantage = advantage_[mini_batch * k:mini_batch * (k + 1), :]
                td_target = td_target_[mini_batch * k:mini_batch * (k + 1), :]

                if self.config.continuous:
                    mu = self.actor(s)
                    action_var = self.action_var.expand_as(mu)
                    cov_mat = torch.diag_embed(action_var).to(
                        self.config.device)
                    dist = MultivariateNormal(mu, cov_mat)
                    log_pi_a = torch.unsqueeze(dist.log_prob(a), 1)
                    entropy = dist.entropy()
                else:
                    pi = self.actor(s)
                    log_pi_a = torch.log(pi.gather(1, a))
                    entropy = Categorical(pi).entropy()

                ratio = torch.exp(log_pi_a - log_prob_a)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.config.eps_clip,
                                    1 + self.config.eps_clip) * advantage

                loss = -torch.min(surr1, surr2) + \
                       self.config.mse_cof * self.MseLoss(td_target.detach(), self.critic(s)) - \
                       self.config.entropy_cof * entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                # nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
