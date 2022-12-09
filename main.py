import numpy as np
import configargparse
import torch
from torch import nn
from torch.distributions import Categorical, MultivariateNormal, Normal
from tensorboardX import SummaryWriter
from env import GymEnvironment
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
from tqdm import tqdm
from agent import PPO

# Global plot settings
sns.set_style("dark")
# sns.despine(left=True)
plt.rcParams["font.family"] = "Times New Roman"


class SADataset(Dataset):

    def __init__(self, buffer_dict):
        self.s_list = []
        self.a_list = []

        keys = list(buffer_dict['s'].keys())

        if len(keys) > 200:
            keys = keys[:200]

        for i in keys:
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

    # for fb
    # model.load_state_dict(
    #     torch.load(
    #         r".\output\FlappyBird\False\2022-10-31-19-53-53\model\model_1202.pkl"
    #     ))
    # buffer_name = 'flappybird_buffer_dict'

    # for ll
    # model.load_state_dict(
    #     torch.load(
    #         r".\output\LunarLander-v2\True\2022-10-31-15-54-19\model\model_14000.pkl"
    #     ))
    # buffer_name = 'lunarLander_buffer_dict'

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
                if ep_reward > 200:
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


def plot_controller(filepath):
    buffer_name = 'cartpole_buffer_dict'
    # buffer_name = 'flappybird_buffer_dict'
    # buffer_name = 'lunarLander_buffer_dict'

    if config.env == 'FlappyBird':
        env = FlappyBirdEnv(seed=config.seed)
        env.step(0)
    else:
        env = GymEnvironment(env_name=config.env,
                             seed=config.seed,
                             delay_step=config.delay_step)
    state_dim, action_dim = env.get_space_dim()

    model = PPO(config, state_dim, action_dim).to(config.device)

    model.load_state_dict(torch.load(filepath))

    if config.env == 'FlappyBird':
        ob_high = [300, 15, 300, 40, 60]
        ob_low = [100, -15, 10, -60, -40]
    elif config.env == 'CartPole-v1':
        ob_high, ob_low = env.env.observation_space.high, env.env.observation_space.low
        ob_high[1], ob_low[1] = 10, -10
        ob_high[3], ob_low[3] = 10, -10
    elif config.env == 'LunarLander-v2':
        ob_high = [1, 1.5, 1, 1, 1, 1, 1, 1]
        ob_low = [-1, 0, -1, -1, -1, -1, 0, 0]

    sns.set_style("dark")
    sns.despine(left=True)
    plt.rcParams["font.family"] = "Times New Roman"

    fig = plt.figure(figsize=(8, 4), tight_layout=True)
    count = 1
    # x_labels = ['CartPosition', 'CartVelocity', 'PoleAngle', 'PoleVelocityAtTip']
    # x_labels = ['Position', 'Velocity', 'DistenceToNext', 'PositionT', 'PositionB']

    for _action in model.actor.rule_dict.keys():
        for _, _rule in enumerate(model.actor.rule_dict[_action]):
            for mfID, _membership_network in enumerate(
                    _rule.membership_network_list):
                fig.add_subplot(len(model.actor.rule_dict.keys()),
                                len(_rule.membership_network_list), count)
                count += 1

                state_id = _rule.state_id[mfID]
                x = torch.linspace(ob_low[state_id], ob_high[state_id], 100)
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

    plt.savefig(f'./tmp/auto_{buffer_name}.pdf')
    plt.show()


def get_ob_high_low(env_name, env):
    if env_name == 'FlappyBird':
        # ob_high = [300, 15, 300, 40, 60]
        # ob_low = [100, -15, 10, -60, -40]
        ob_high = [300, 2, 300, 10, 10]
        ob_low = [100, -2, 10, -10, -10]
    elif 'CartPole' in env_name:
        ob_high, ob_low = env.env.observation_space.high, env.env.observation_space.low
        ob_high[1], ob_low[1] = 10, -10
        ob_high[3], ob_low[3] = 10, -10
    elif 'LunarLander' in env_name:
        ob_high = [1, 1.5, 1, 1, 1, 1, 1, 1]
        ob_low = [-1, 0, -1, -1, -1, -1, 0, 0]

    return ob_high, ob_low


def plot_rule(filepath, rule_id=0):
    if config.env == 'FlappyBird':
        env = FlappyBirdEnv(seed=config.seed)
        env.step(0)
    else:
        env = GymEnvironment(env_name=config.env,
                             seed=config.seed,
                             delay_step=config.delay_step)
    state_dim, action_dim = env.get_space_dim()

    model = PPO(config, state_dim, action_dim).to(config.device)
    model.load_state_dict(torch.load(filepath))

    leaves_distribution = torch.softmax(model.policy.param, 1)
    print('leaves_distribution: {}'.format(leaves_distribution))

    ob_high, ob_low = get_ob_high_low(config.env, env)

    plt.figure(figsize=(8, 2), tight_layout=True)
    count = 1
    # x_labels = ['CartPosition', 'CartVelocity', 'PoleAngle', 'PoleVelocityAtTip']
    # x_labels = ['Position', 'Velocity', 'DistenceToNext', 'PositionT', 'PositionB']

    _rule = model.policy.rule_tree[rule_id]

    for mfID, _membership_network in enumerate(_rule.membership_network_list):
        # fig.add_subplot(1, len(_rule.membership_network_list), count)
        plt.subplot(1, len(_rule.membership_network_list), count)
        count += 1

        state_id = _rule.state_id[mfID]
        x = torch.linspace(ob_low[state_id], ob_high[state_id], 100)
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

    plt.show()


def evaluate(model):
    if config.env == 'FlappyBird':
        env = FlappyBirdEnv(seed=config.seed, display_screen=True)
        env.step(0)
    else:
        env = GymEnvironment(env_name=config.env,
                             seed=config.seed,
                             delay_step=config.delay_step)

    score = 0.0
    n = 10

    for i in range(n):
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

            # env.env.render()
            # time.sleep(1 / 10)
            # print(s)

            s_lst.append(s)
            a_lst.append(a)

            epi_reward += r
            score += r
            s = s_prime

        print('epi_reward {}: {}'.format(i, epi_reward))
    return score / n


def train_using_buffer():
    wandb.init(
        project='knowledge-transfer',
        name=f'seed_{config.seed}',
        group=f'{config.env}_{config.no_controller}_{config.info}',
        dir='./output',
    )

    buffer_name = 'cartpole_buffer_dict'
    # buffer_name = 'flappybird_buffer_dict'
    # buffer_name = 'lunarLander_buffer_dict'

    with open('./buffer/' + buffer_name + '.pkl', 'rb') as f:
        buffer_dict = pickle.load(f)

    sa_dataset = SADataset(buffer_dict)
    train_loader = DataLoader(dataset=sa_dataset, batch_size=64, shuffle=True)

    if config.env == 'FlappyBird':
        env = FlappyBirdEnv(seed=config.seed)
        env.step(0)
    else:
        env = GymEnvironment(env_name=config.env,
                             seed=config.seed,
                             delay_step=config.delay_step)
    state_dim, action_dim = env.get_space_dim()

    model = PPO(config, state_dim, action_dim).to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.actor.parameters(), lr=1e-4)

    best = -np.inf

    for epoch in range(1000):
        losses = []

        for _, data in tqdm(enumerate(train_loader)):
            s, a = data

            s = Variable(s)
            a = Variable(a)

            outputs = model.actor(s)

            optimizer.zero_grad()
            loss = criterion(outputs, a.long()).mean()
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
            optimizer.step()

        res = evaluate(model)
        print("epoch: {}, ave score: {:.1f}, loss: {:.3f}".format(
            epoch, res, np.mean(losses)))
        wandb.log({'reward': res})

        if res > best:
            torch.save(
                model.state_dict(),
                os.path.join('./buffer', buffer_name + '_model_best.pkl'))
            best = res
            print('saved')


def discretization_v1(model):
    for i, rule in enumerate(model.rule_tree):
        print(i, torch.softmax(rule.p_select, 1))
    print('---')
    print(torch.softmax(model.param, 1))

    print('Discretization...')

    threshold_gumbel = 0.8
    
    # action distribution
    probs = torch.softmax(model.param, 1).detach().numpy()
    log_probs = np.log2(probs)
    entropy = -1 * np.sum(probs * log_probs, axis=1)
    print('action entropy: {}'.format(entropy))

    one_hot_probability = torch.zeros_like(model.param)
    vs, ids = torch.max(model.param, 1)
    one_hot_probability.scatter_(1, ids.view(-1, 1), 1.)
    model.param = nn.Parameter(one_hot_probability * 100)

    # node gumbel softmax
    for i, rule in enumerate(model.rule_tree):
        probs = torch.softmax(rule.p_select, 1).detach().numpy()
        log_probs = np.log2(probs)
        entropy = -1 * np.sum(probs * log_probs, axis=1)

        disable_gumbel = []
        for k in range(len(entropy)):
            if entropy[k] > threshold_gumbel:
                disable_gumbel.append(k)
                print('cut {}, {}'.format(i, k))

        one_hot_probability = torch.zeros_like(rule.p_select)
        vs, ids = torch.max(rule.p_select, 1)
        one_hot_probability.scatter_(1, ids.view(-1, 1), 1.)

        one_hot_probability[disable_gumbel, :] = torch.tensor(
            [1, 0], dtype=torch.float)
        rule.p_select = nn.Parameter(one_hot_probability * 100)

    for i, rule in enumerate(model.rule_tree):
        print(i, torch.softmax(rule.p_select, 1))
    print('---')
    print(torch.softmax(model.param, 1))


def discretization_v3(model):
    print(torch.softmax(model.param, 1))

    print('Discretization...')

    # action distribution
    probs = torch.softmax(model.param, 1).detach().numpy()
    log_probs = np.log2(probs)
    entropy = -1 * np.sum(probs * log_probs, axis=1)
    print('action entropy: {}'.format(entropy))

    one_hot_probability = torch.zeros_like(model.param)
    vs, ids = torch.max(model.param, 1)
    one_hot_probability.scatter_(1, ids.view(-1, 1), 1.)
    model.param = nn.Parameter(one_hot_probability * 100)

    print(torch.softmax(model.param, 1))


def dynamic_demo(filepath):
    # x_labels = ['CartPosition', 'CartVelocity', 'PoleAngle', 'PoleVelocityAtTip']  #* cart
    x_labels = ['X', 'Y', 'VX', 'VY', 'A', 'AV', 'LC', 'RC']  #* lunar
    # x_labels = ['Position', 'Velocity', 'DistenceToNext', 'Position - PositionT', 'Position - PositionB']  #* flappy

    plt.ion()

    def fig_plot(_rule, fig):
        count = 1
        axes = []

        for mfID, _membership_network in enumerate(_rule.membership_network_list):
            ax = fig.add_subplot(1, len(_rule.membership_network_list), count)
            axes.append(ax)

            count += 1

            state_id = _rule.state_id[mfID]
            x = torch.linspace(ob_low[state_id], ob_high[state_id], 100)
            y = torch.zeros_like(x)

            for i, _ in enumerate(x):
                y[i] = _membership_network(
                    _.reshape(-1, 1).to(config.device)).item()

            ax.plot(x.cpu().numpy(), y.cpu().numpy(), linewidth=3)
            # point, = ax.plot(state[mfID], _membership_network(torch.unsqueeze(torch.tensor(state[mfID]), 0).to(torch.float32)).detach().numpy(), marker='o', color='red')
            # points.append(point)

            plt.box(True)
            plt.grid(axis='y')
            plt.ylim([-0.05, 1.05])
            plt.xlabel(x_labels[mfID], fontsize=12)

        return axes

    def fig_plot_point(ax, _rule, state):
        points = []

        for i in range(len(state)):
            point, = ax[i].plot(state[i], _rule.membership_network_list[i](torch.unsqueeze(
                torch.tensor(state[i]), 0).to(torch.float32)).detach().numpy(), marker='o', color='red')
            points.append(point)

        return points

    def fig_plot_membership_functions(model):
        plt.figure(figsize=(16, 8))
        plt.ioff()

        membership_functions = model.rule_tree
        row = len(model.state_components)
        col = 2 ** (len(model.state_components)-1)

        for i, state_component in enumerate(model.state_components):
            for j, k in enumerate(range(2**(i)-1, 2**(i+1)-1)):
                plt.subplot(row, col, i * col + j + 1)
                
                x = torch.linspace(ob_low[state_component], ob_high[state_component], 100)
                y = torch.zeros_like(x)

                for ii, _ in enumerate(x):
                    y[ii] = membership_functions[k](
                        _.reshape(-1, 1).to(config.device)).item()
                
                plt.plot(x.cpu().numpy(), y.cpu().numpy(), linewidth=3)
                plt.box(True)
                plt.grid(axis='y')
                plt.ylim([-0.05, 1.05])
                plt.xlabel(x_labels[state_component])
        
        plt.tight_layout()
        plt.show()


    if config.env == 'FlappyBird':
        env = FlappyBirdEnv(seed=config.seed, display_screen=True)
        env.step(0)
    else:
        env = GymEnvironment(env_name=config.env,
                             seed=config.seed, delay_step=config.delay_step)

    state_dim, action_dim = env.get_space_dim()

    ob_high, ob_low = get_ob_high_low(config.env, env)

    model = PPO(config, state_dim, action_dim).to(config.device)
    model.load_state_dict(torch.load(filepath))

    torch.set_printoptions(precision=2, sci_mode=False)
    discretization_v3(model.policy)  #*
    
    print(evaluate(model))
    # print(model.policy.leaf_use_count / np.sum(model.policy.leaf_use_count))
    exit()

    fig_plot_membership_functions(model.policy)
    exit()

    rule_tree = model.policy.rule_tree
    depth = model.policy.depth

    figs_num = 2 ** depth - 1

    figs = []
    axes = []

    for i in range(figs_num):
        figs.append(plt.figure(figsize=(8, 1.5), tight_layout=True))
        axes.append(fig_plot(rule_tree[i], figs[-1]))

    score = 0.0
    n = 10

    plt.show()

    for i in range(n):
        s = env.reset()
        done = False
        epi_reward = 0
        s_lst = []
        a_lst = []

        while not done:
            prob = model.actor(torch.as_tensor(
                s.reshape(1, -1), dtype=torch.float).to(config.device))
            m = Categorical(prob)
            a = m.sample().item()

            try:
                [[point.remove() for point in points] for points in points_all]
            except NameError:
                pass

            # points_all = []
            # for i in range(len(figs)):
            #     points_all.append(fig_plot_point(axes[i], rule_tree[i], s))
            
            print(s, a)
            env.env.render()
            s_prime, r, done, _ = env.step(a)

            s_lst.append(s)
            a_lst.append(a)

            epi_reward += r
            score += r
            s = s_prime

        print('epi_reward {}: {}'.format(i, epi_reward))


def train(source_filepath=None):
    wandb.init(
        project='knowledge-transfer',
        name=f'seed_{config.seed}',
        group=f'{config.env}_{config.no_controller}_{config.info}',
        dir='./output',
    )

    render = False
    # writer = SummaryWriter(log_dir=log_dir)

    if config.env == 'FlappyBird':
        env = FlappyBirdEnv(seed=config.seed, display_screen=False)
        env.step(0)
    else:
        env = GymEnvironment(env_name=config.env,
                             seed=config.seed, delay_step=config.delay_step)

    state_dim, action_dim = env.get_space_dim()
    model = PPO(config, state_dim, action_dim,
                source_filepath).to(config.device)
    
    # pretrained_dict = torch.load(r'tmp/model/cart-model_best.pkl')
    # net_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict.keys())}
    # net_dict.update(pretrained_dict)
    # model.load_state_dict(net_dict)

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug('%s: %s' % (name, param.size()))

    update_count = 0
    update_count_eq = 0
    ep_count = 0
    ep_reward = 0
    last_ep_reward = 0
    best = -np.inf

    s = env.reset()
    steps = 0
    running_reward_list = []

    while True:
        done = False

        while not done:
            for _ in range(config.t_horizon):
                with torch.no_grad():
                    if config.continuous:
                        mu = model.actor(
                            torch.as_tensor(s.reshape(1, -1),
                                            dtype=torch.float).to(
                                                config.device))
                        action_var = model.action_var.expand_as(mu)
                        cov_mat = torch.diag_embed(action_var).to(
                            config.device)
                        dist = MultivariateNormal(mu, cov_mat)
                        a = dist.sample()
                        a = torch.clamp(a, -config.action_scale,
                                        config.action_scale)
                        logprob = dist.log_prob(a)
                        a = a.cpu().data.numpy().flatten()
                    else:
                        prob = model.actor(
                            torch.as_tensor(s.reshape(1, -1),
                                            dtype=torch.float).to(
                                                config.device), ep_count)  #* ep_count
                        m = Categorical(prob)
                        a = m.sample().item()
                        logprob = torch.log(torch.squeeze(prob)[a])

                s_prime, r, done, _ = env.step(a)
                ep_reward += r

                model.put_data((s, a, r / config.reward_scale, s_prime, logprob.item(), done))
                s = s_prime

                steps += 1
                update_count_eq = steps // config.t_horizon

                if done:
                    ep_count += 1

                    # writer.add_scalar('reward/ep_reward', ep_reward, ep_count)
                    wandb.log({
                        'ep_reward': ep_reward,
                        'ep_count': ep_count,
                        'steps': steps,
                        'update_count_eq': update_count_eq
                    })
                    # last_ep_reward = ep_reward
                    running_reward_list.append(ep_reward)
                    ep_reward = 0
                    s = env.reset()

                    break

            model.train_net()
            update_count += 1

        if ep_count % config.print_interval == 0:
            running_reward = np.mean(np.array(running_reward_list)[-10:])

            logger.info(
                "episode: {}, update count: {}, running reward: {:.1f}, steps:{}"
                .format(ep_count, update_count_eq, running_reward, steps))

            if running_reward > best:
                best = running_reward
                torch.save(model.state_dict(), os.path.join(
                    model_dir, 'model_best.pkl'))
                logger.info('Saved')

        if ep_count % config.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(
                model_dir, 'model_{}.pkl'.format(ep_count)))

        # writer.add_scalar('reward/update_reward', last_ep_reward, update_count)

        if ep_count >= config.max_update:
            break

    torch.save(model.state_dict(), os.path.join(
        model_dir, 'model_{}.pkl'.format(ep_count)))
    env.close()

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
                   default=10000,
                   help='Max episode count for training')
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
    log_dir = os.path.join(config.output_dir, config.env, 'Auto', time_stamp, 'log')
    model_dir = os.path.join(config.output_dir, config.env, 'Auto', time_stamp, 'model')

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
        f.write('\n'.join(
            ["%s: %s" % (key, value) for key, value in vars(config).items()]))

    for k in list(vars(config).keys()):
        logger.debug('%s: %s' % (k, vars(config)[k]))

    apply_seed(config.seed)

    # * choose an entry
    # train()

    # train(r'tmp/model/cart-model_best.pkl')
    train(r'tmp/model/cart-tree-v3-s0123-model_best.pkl')

    # train(r'tmp\model\flappy-model_best.pkl')
    # train(r'tmp/model/flappy-tree-v3-s134-model_best.pkl')

    # train(r'tmp/model/lunar-tree-v3-s2345-model_best.pkl')
    # train(r'tmp/model/lunar-model_best.pkl')

    # collect_buffer()
    # train_using_buffer()

    # dynamic_demo(r'tmp\model\cart-tree-v3-s0123-model_best.pkl')
    # dynamic_demo(r'tmp\model\cart-model_best.pkl')

    # dynamic_demo(r'tmp\model\lunar-tree-v3-s2345-model_best.pkl')
    # dynamic_demo(r'tmp\model\lunar-tree-v3-sall-model_best.pkl')
    # dynamic_demo(r'tmp\model\lunar-model_best.pkl')

    # dynamic_demo(r'tmp\model\flappy-model_best.pkl')
    # dynamic_demo(r'tmp\model\flappy-tree-v3-s134-model_best.pkl')
