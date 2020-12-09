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

    total_r = 0
    training_count = 0

    for epi in range(config.max_episode):
        s = env.reset()
        done = False

        while not done:
            for t in range(config.t_horizon):
                if config.no_controller:
                    prob = model.pi(torch.from_numpy(s).float().cuda())
                else:
                    prob = torch.squeeze(model.controller(torch.from_numpy(s).float().reshape(1, -1).cuda()))
                # print(prob)

                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)

                model.put_data((s, a, r / config.reward_reduction_ratio, s_prime, prob[a].item(), done))
                s = s_prime

                total_r += r
                if done:
                    break

            model.train_net()
            training_count += config.k_epoch

        if epi % config.print_interval == 0 and epi != 0:
            print("episode: {}, training count: {}, avg reward: {:.1f}".format(epi, training_count, total_r / config.print_interval))
            writer.add_scalar('avg_reward', total_r / config.print_interval, training_count)
            total_r = 0

    env.close()


if __name__ == '__main__':
    p = configargparse.ArgumentParser()

    p.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file')

    p.add_argument('--env', type=str, default='CartPole-v1', help='The name of environment')
    p.add_argument('--seed', type=int, default=0, help='Seed for environment')
    p.add_argument('--delay_step', type=int, default=1, help='Delay step for environment')

    p.add_argument('--reward_reduction_ratio', type=int, default=100, help='The ratio of reward reduction')
    p.add_argument('--max_episode', type=int, default=5000, help='Max episode for training')
    p.add_argument('--k_epoch', type=int, default=1, help='Epoch per training')
    p.add_argument('--t_horizon', type=int, default=128, help='Max horizon per training')

    p.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    p.add_argument('--gamma', type=float, default=0.99, help='Discount of reward')
    p.add_argument('--lmbda', type=float, default=0.95, help='Discount of GAE')
    p.add_argument('--eps_clip', type=float, default=0.2, help='Clip epsilon of PPO')

    p.add_argument('--print_interval', type=int, default=10, help='Print interval during training')
    p.add_argument('--no_controller', default=False, action='store_true', help='Whether to use the controller')

    config = p.parse_args()

    train()
