import numpy as np
import configargparse
import torch
from torch import nn
from torch.distributions import Categorical, MultivariateNormal, Normal
from tensorboardX import SummaryWriter
from env import GymEnvironment
from agent import PPO
import os
from logger import logger
from env_flappybird_kogun import FlappyBirdEnv
import time
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from torch.utils.data import Dataset, DataLoader
import pickle
from torch import optim, nn
from torch.autograd import Variable


class SADataset(Dataset):

    def __init__(self, buffer_dict):
        self.s_list = []
        self.a_list = []

        for i in range(len(buffer_dict['s'])):
            for j in range(len(buffer_dict['s'][i])):
                self.s_list.append(buffer_dict['s'][i][j])
                self.a_list.append(buffer_dict['a'][i][j])

        self.s_list = torch.from_numpy(np.array(self.s_list)).float()
        self.a_list = torch.from_numpy(np.array(self.a_list)).float()

    def __getitem__(self, index):
        return self.s_list[index, :], self.a_list[index]

    def __len__(self):
        assert len(self.s_list) == len(self.a_list)
        return len(self.s_list)


def apply_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_buffer(n=1000):
    if config.env == 'FlappyBird':
        env = FlappyBirdEnv(seed=config.seed)
        env.step(0)
    else:
        env = GymEnvironment(env_name=config.env,
                             seed=config.seed,
                             delay_step=config.delay_step)
    state_dim, action_dim = env.get_space_dim()

    model = PPO(config, state_dim, action_dim).to(config.device)

    # for cp
    model.load_state_dict(
        torch.load(
            r".\output\CartPole-v1\True\2022-10-20-23-00-29\model\model_4000.pkl"
        ))
    buffer_name = 'cartpole_buffer_dict'

    ep_reward = 0
    s = env.reset()

    buffer_dict = {'s': {}, 'a': {}}

    for epi in range(n):
        s_lst = []
        a_lst = []

        while True:
            with torch.no_grad():
                if config.continuous:
                    mu = model.actor(
                        torch.as_tensor(s.reshape(1, -1),
                                        dtype=torch.float).to(config.device))
                    action_var = model.action_var.expand_as(mu)
                    cov_mat = torch.diag_embed(action_var).to(config.device)
                    dist = MultivariateNormal(mu, cov_mat)
                    a = dist.sample()
                    a = torch.clamp(a, -config.action_scale,
                                    config.action_scale)
                    logprob = dist.log_prob(a)
                    a = a.cpu().data.numpy().flatten()
                else:
                    prob = model.actor(
                        torch.as_tensor(s.reshape(1, -1),
                                        dtype=torch.float).to(config.device))
                    m = Categorical(prob)
                    a = m.sample().item()
                    logprob = torch.log(torch.squeeze(prob)[a])

            s_prime, r, done, _ = env.step(a)
            s_lst.append(s)
            a_lst.append(a)

            ep_reward += r

            # env.render()

            s = s_prime

            if done:
                buffer_dict['s'][epi] = s_lst
                buffer_dict['a'][epi] = a_lst

                print('epi: {} | ep_reward: {}'.format(epi, ep_reward))
                ep_reward = 0
                s = env.reset()
                done = False

                break

    with open('./buffer/' + buffer_name + '.pkl', 'wb') as f:
        pickle.dump(buffer_dict, f, pickle.HIGHEST_PROTOCOL)

    env.close()


def train():
    wandb.init(
        project='knowledge-rl',
        name=f'seed_{config.seed}',
        group=f'{config.env}_Auto_{config.info}',
        dir='./output',
    )

    render = False
    plot = False

    writer = SummaryWriter(log_dir=log_dir)

    if config.env == 'FlappyBird':
        env = FlappyBirdEnv(seed=config.seed)
        env.step(0)
    else:
        env = GymEnvironment(env_name=config.env,
                             seed=config.seed,
                             delay_step=config.delay_step)
    state_dim, action_dim = env.get_space_dim()

    model = PPO(config, state_dim, action_dim).to(config.device)

    if plot:
        model.load_state_dict(torch.load(r".\model_30000.pkl"))

        ob_high, ob_low = env.env.observation_space.high, env.env.observation_space.low
        # ob_high[1], ob_low[1] = 10, -10
        # ob_high[3], ob_low[3] = 10, -10
        sns.set_style("dark")
        sns.despine(left=True)
        plt.rcParams["font.family"] = "Times New Roman"

        fig = plt.figure(figsize=(8, 4), tight_layout=True)
        count = 1
        # x_labels = ['CartPosition', 'CartVelocity', 'PoleAngle', 'PoleVelocityAtTip']

        for _action in model.actor.rule_dict.keys():
            for ruleID, _rule in enumerate(model.actor.rule_dict[_action]):
                for mfID, _membership_network in enumerate(
                        _rule.membership_network_list):
                    fig.add_subplot(len(model.actor.rule_dict.keys()),
                                    len(_rule.membership_network_list), count)
                    count += 1

                    state_id = _rule.state_id[mfID]
                    x = torch.linspace(ob_low[state_id], ob_high[state_id],
                                       100)
                    y = torch.zeros_like(x)

                    for i, _ in enumerate(x):
                        y[i] = _membership_network(
                            _.reshape(-1, 1).to(config.device)).item()

                    plt.plot(x.cpu().numpy(), y.cpu().numpy(), linewidth=3)
                    plt.box(True)
                    plt.grid(axis='y')
                    plt.ylim([-0.05, 1.05])
                    # plt.xlabel(x_labels[mfID], fontsize=12)
                    if mfID == 0:
                        plt.ylabel('Membership', fontsize=12)
        plt.savefig('ll_auto.pdf')
        # plt.show()
        exit(0)

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug('%s: %s' % (name, param.size()))

    update_count = 0
    update_count_eq = 0
    ep_count = 0
    ep_reward = 0
    last_ep_reward = 0

    s = env.reset()
    steps = 0

    while True:
        for _ in range(config.t_horizon**10):  # TODO
            with torch.no_grad():
                if config.continuous:
                    mu = model.actor(
                        torch.as_tensor(s.reshape(1, -1),
                                        dtype=torch.float).to(config.device))
                    action_var = model.action_var.expand_as(mu)
                    cov_mat = torch.diag_embed(action_var).to(config.device)
                    dist = MultivariateNormal(mu, cov_mat)
                    a = dist.sample()
                    a = torch.clamp(a, -config.action_scale,
                                    config.action_scale)
                    logprob = dist.log_prob(a)
                    a = a.cpu().data.numpy().flatten()
                else:
                    prob = model.actor(
                        torch.as_tensor(s.reshape(1, -1),
                                        dtype=torch.float).to(config.device))
                    m = Categorical(prob)
                    a = m.sample().item()
                    logprob = torch.log(torch.squeeze(prob)[a])

            s_prime, r, done, _ = env.step(a)
            ep_reward += r

            # env.render()

            if render:
                env.render()
                logger.debug([s_prime, a, r])
                time.sleep(1 / 10)

            model.put_data(
                (s, a, r / config.reward_scale, s_prime, logprob.item(), done))
            s = s_prime

            steps += 1
            if steps % config.t_horizon == 0:
                update_count_eq += 1

            if done:
                ep_count += 1
                writer.add_scalar('reward/ep_reward', ep_reward, ep_count)
                wandb.log({
                    'ep_reward': ep_reward,
                    'ep_count': ep_count,
                    'steps': steps,
                    'update_count_eq': steps // config.t_horizon
                })
                last_ep_reward = ep_reward
                ep_reward = 0
                s = env.reset()
                done = False

                break

        model.train_net()
        update_count += 1

        if update_count % config.print_interval == 0:
            if config.no_controller:
                logger.info(
                    "episode: {}, update count: {}, reward: {:.1f}, steps:{}".
                    format(ep_count, update_count, last_ep_reward, steps))
            else:
                logger.info(
                    "episode: {}, update count: {}, reward: {:.1f}, steps:{}".
                    format(ep_count, update_count, last_ep_reward, steps))

        if update_count % config.save_interval == 0:
            torch.save(
                model.state_dict(),
                os.path.join(model_dir, 'model_{}.pkl'.format(update_count)))

        writer.add_scalar('reward/update_reward', last_ep_reward, update_count)

        if update_count_eq >= config.max_update:  # TODO
            break

    env.close()


def evaluate(model):
    env = GymEnvironment(env_name=config.env,
                         seed=config.seed,
                         delay_step=config.delay_step)

    score = 0.0
    n = 10

    for epi in range(n):
        s = env.reset()
        done = False
        epi_reward = 0
        s_lst = []
        a_lst = []

        while not done:
            prob = model.actor(
                torch.as_tensor(s.reshape(1, -1),
                                dtype=torch.float).to(config.device))
            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done, info = env.step(a)
            s_lst.append(s)
            a_lst.append(a)

            epi_reward += r
            score += r
            s = s_prime

    return score / n


def train_using_buffer():
    buffer_name = 'cartpole_buffer_dict'

    with open('./buffer/' + buffer_name + '.pkl', 'rb') as f:
        buffer_dict = pickle.load(f)

    sa_dataset = SADataset(buffer_dict)
    train_loader = DataLoader(dataset=sa_dataset, batch_size=128, shuffle=True)

    env = GymEnvironment(env_name=config.env,
                         seed=config.seed,
                         delay_step=config.delay_step)

    state_dim, action_dim = env.get_space_dim()
    model = PPO(config, state_dim, action_dim).to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.actor.parameters(), lr=1e-4)

    best = -np.inf

    for epoch in range(1000):
        for _, data in enumerate(train_loader):
            s, a = data

            s = Variable(s)
            a = Variable(a)

            outputs = model.actor(s)

            optimizer.zero_grad()
            loss = criterion(outputs, a.long())
            loss.backward()
            nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
            optimizer.step()

        res = evaluate(model)
        print("epoch: {}, ave score: {:.1f}".format(epoch, res))

        if res > best:
            torch.save(
                model.state_dict(),
                os.path.join('./buffer', buffer_name + '_model_best.pkl'))
            best = res
            print('saved')


if __name__ == '__main__':
    p = configargparse.ArgumentParser()

    p.add_argument('-c',
                   '--config_filepath',
                   required=False,
                   is_config_file=True,
                   help='Path to config file')
    p.add_argument('--device', type=str, default='cpu', help='cpu or cuda:0')
    p.add_argument('--output_dir',
                   type=str,
                   default='./output/',
                   help='The name of environment')
    p.add_argument('--continuous',
                   default=False,
                   action='store_true',
                   help='Whether to use continuous action space')
    p.add_argument('--env',
                   type=str,
                   default='CartPole-v1',
                   help='The name of environment')
    p.add_argument('--seed', type=int, default=0, help='Seed for reproducible')
    p.add_argument('--delay_step',
                   type=int,
                   default=1,
                   help='Delay step for environment')
    p.add_argument('--reward_scale',
                   type=int,
                   default=100.0,
                   help='The ratio of reward reduction')
    p.add_argument('--max_update',
                   type=int,
                   default=4000,
                   help='Max update count for training')
    p.add_argument('--save_interval',
                   type=int,
                   default=1000,
                   help='The save interval during training')
    p.add_argument('--k_epoch', type=int, default=3, help='Epoch per training')
    p.add_argument('--t_horizon',
                   type=int,
                   default=128,
                   help='Max horizon per training')
    p.add_argument('--learning_rate',
                   type=float,
                   default=1e-4,
                   help='Learning rate for training')
    p.add_argument('--gamma',
                   type=float,
                   default=0.98,
                   help='Discount of reward')
    p.add_argument('--lmbda', type=float, default=0.95, help='Discount of GAE')
    p.add_argument('--eps_clip',
                   type=float,
                   default=0.2,
                   help='Clip epsilon of PPO')
    p.add_argument('--p_cof',
                   type=float,
                   default=0.7,
                   help='The coefficient of the start of p')
    p.add_argument('--p_cof_end',
                   type=float,
                   default=0.1,
                   help='The coefficient of the end of p')
    p.add_argument('--p_total_step',
                   type=float,
                   default=2000,
                   help='The total step of p')
    p.add_argument('--std',
                   type=float,
                   default=0.5,
                   help='The value of std for continuous PPO')
    p.add_argument('--no_gae',
                   default=False,
                   action='store_true',
                   help='Whether to use GAE for PPO')
    p.add_argument('--entropy_cof',
                   type=float,
                   default=0,
                   help='The entropy coefficient for PPO')
    p.add_argument('--mse_cof',
                   type=float,
                   default=1,
                   help='The MSE coefficient for PPO')
    p.add_argument('--action_scale',
                   type=float,
                   default=1,
                   help='The scale for action')
    p.add_argument('--print_interval',
                   type=int,
                   default=20,
                   help='Print interval during training')
    p.add_argument('--no_controller',
                   default=False,
                   action='store_true',
                   help='Whether to use the controller')
    p.add_argument('--log_extra',
                   default=False,
                   action='store_true',
                   help='Whether to log extra information')
    p.add_argument('--log_extra_interval',
                   type=int,
                   default=1000,
                   help='The log interval during training')
    p.add_argument('--use_minibatch',
                   default=False,
                   action='store_true',
                   help='Whether to use minibatch')
    p.add_argument('--minibatch',
                   type=int,
                   default=30,
                   help='The number of minibatch')
    p.add_argument('--info', type=str, default='')

    config = p.parse_args()

    time_stamp = time.strftime("%F-%H-%M-%S")
    log_dir = os.path.join(config.output_dir, config.env, 'Auto', time_stamp,
                           'log')
    model_dir = os.path.join(config.output_dir, config.env, 'Auto', time_stamp,
                             'model')

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
        f.write('\n'.join(
            ["%s: %s" % (key, value) for key, value in vars(config).items()]))

    for k in list(vars(config).keys()):
        logger.debug('%s: %s' % (k, vars(config)[k]))

    apply_seed(config.seed)

    # train()
    # collect_buffer()
    train_using_buffer()
