import configargparse
import torch
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from env import Environment
from agent import PPO


def train():
    writer = SummaryWriter()

    env = Environment(env_name=config.env, seed=config.seed, delay_step=config.delay_step)
    config.action_dim, config.state_dim = env.get_space_dim()
    model = PPO(config).cuda()

    print("-----All trainable tensor-----")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())

    update_count = 0
    ep_count = 0
    ep_reward = 0
    last_ep_reward = 0

    s = env.reset()
    while True:
        for i in range(config.t_horizon):
            if config.no_controller:
                prob = model.pi(torch.from_numpy(s).float().cuda())
            else:
                prob = torch.squeeze(model.controller(torch.from_numpy(s).float().reshape(1, -1).cuda()))

            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done, info = env.step(a)
            ep_reward += r

            model.put_data((s, a, r / config.reward_reduction_ratio, s_prime, prob[a].item(), done))
            s = s_prime

            if done:
                ep_count += 1
                writer.add_scalar('reward/epi_reward', ep_reward, ep_count)
                last_ep_reward = ep_reward
                ep_reward = 0
                s = env.reset()

        model.train_net()
        update_count += 1

        if update_count % config.print_interval == 0:
            print("episode: {}, update count: {}, reward: {:.1f}".format(ep_count, update_count, last_ep_reward))

            if config.no_controller:
                for i, (name, param) in enumerate(model.fc_pi.named_parameters()):
                    writer.add_histogram('fc_pi/' + name, param, update_count)
            else:
                # TODO: abaabaaba
                for i, (name, param) in enumerate(model.controller.rule_dict[0][0].membership_network_list[0].named_parameters()):
                    writer.add_histogram('{}/{}'.format(model.controller.rule_dict[0][0].id, name), param, update_count)

            for i, (name, param) in enumerate(model.fc1.named_parameters()):
                writer.add_histogram('fc1/' + name, param, update_count)

            for i, (name, param) in enumerate(model.fc_v.named_parameters()):
                writer.add_histogram('fc_v/' + name, param, update_count)

        writer.add_scalar('reward/update_reward', last_ep_reward, update_count)

        if update_count >= config.max_update_count:
            break

    env.close()


if __name__ == '__main__':
    p = configargparse.ArgumentParser()

    p.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file')

    p.add_argument('--env', type=str, default='CartPole-v1', help='The name of environment')
    p.add_argument('--seed', type=int, default=0, help='Seed for environment')
    p.add_argument('--delay_step', type=int, default=1, help='Delay step for environment')

    p.add_argument('--reward_reduction_ratio', type=int, default=100, help='The ratio of reward reduction')
    # p.add_argument('--max_episode', type=int, default=5000, help='Max episode for training')
    p.add_argument('--max_update_count', type=int, default=25000, help='Max update count for training')
    p.add_argument('--k_epoch', type=int, default=3, help='Epoch per training')
    p.add_argument('--t_horizon', type=int, default=128, help='Max horizon per training')

    p.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    p.add_argument('--gamma', type=float, default=0.99, help='Discount of reward')
    p.add_argument('--lmbda', type=float, default=0.95, help='Discount of GAE')
    p.add_argument('--eps_clip', type=float, default=0.2, help='Clip epsilon of PPO')

    p.add_argument('--print_interval', type=int, default=20, help='Print interval during training')
    p.add_argument('--no_controller', default=False, action='store_true', help='Whether to use the controller')

    config = p.parse_args()

    print("-----All arguments-----")
    for k in list(vars(config).keys()):
        print('%s: %s' % (k, vars(config)[k]))

    train()
