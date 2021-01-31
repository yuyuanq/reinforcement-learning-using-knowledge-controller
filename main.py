import numpy as np
import configargparse
import torch
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from env import GymEnvironment
from agent import PPO
import os
from logger import logger
from env_flappybird import FlappyBirdEnv
import time


def apply_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    writer = SummaryWriter(log_dir=log_dir)

    if config.env == 'FlappyBird':
        env = FlappyBirdEnv(seed=config.seed, display_screen=False)
    else:
        env = GymEnvironment(env_name=config.env, seed=config.seed, delay_step=config.delay_step)
    state_dim, action_dim = env.get_space_dim()
    model = PPO(config, state_dim, action_dim).cuda()

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug('%s: %s' % (name, param.size()))

    update_count = 0
    ep_count = 0
    ep_reward = 0
    last_ep_reward = 0

    s = env.reset()
    while True:
        for i in range(config.t_horizon):
            prob = model.actor(torch.unsqueeze(torch.from_numpy(s), 0).float().cuda())

            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done, info = env.step(a)
            ep_reward += r

            model.put_data((s, a, r / config.reward_ratio, s_prime, torch.squeeze(prob)[a].item(), done))
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
            if config.no_controller:
                logger.info(
                    "episode: {}, update count: {}, reward: {:.1f}".format(ep_count, update_count, last_ep_reward))
            else:
                logger.info(
                    "episode: {}, update count: {}, reward: {:.1f}, p_cof: {:.2f}".
                        format(ep_count, update_count, last_ep_reward, model.actor.p_cof))

        # if config.log_extra and (update_count % config.log_extra_interval == 0 or update_count == 1):
        #     if not config.no_controller:
        #         plot_mf(model, ob_low, ob_high, writer, update_count)

        if update_count % config.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, 'model_{}.pkl'.format(update_count)))

        writer.add_scalar('reward/update_reward', last_ep_reward, update_count)

        if update_count >= config.max_update:
            break

    env.close()


if __name__ == '__main__':
    p = configargparse.ArgumentParser()

    p.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file')
    p.add_argument('--output_dir', type=str, default='./output/', help='The name of environment')

    p.add_argument('--env', type=str, default='CartPole-v1', help='The name of environment')
    p.add_argument('--seed', type=int, default=0, help='Seed for reproducible')
    p.add_argument('--delay_step', type=int, default=1, help='Delay step for environment')
    p.add_argument('--reward_ratio', type=int, default=100, help='The ratio of reward reduction')
    p.add_argument('--max_update', type=int, default=10000, help='Max update count for training')
    p.add_argument('--save_interval', type=int, default=5000, help='The save interval during training')
    p.add_argument('--k_epoch', type=int, default=3, help='Epoch per training')
    p.add_argument('--t_horizon', type=int, default=128, help='Max horizon per training')
    p.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    p.add_argument('--gamma', type=float, default=0.99, help='Discount of reward')
    p.add_argument('--lmbda', type=float, default=0.95, help='Discount of GAE')
    p.add_argument('--eps_clip', type=float, default=0.2, help='Clip epsilon of PPO')
    p.add_argument('--p_cof', type=float, default=0.7, help='The coefficient of the start of p')
    p.add_argument('--p_cof_end', type=float, default=0.1, help='The coefficient of the end of p')
    p.add_argument('--p_total_step', type=float, default=2000, help='The total step of p')
    p.add_argument('--use_reward_normalization', default=False, action='store_true',
                   help='Whether to use the reward normalization')
    p.add_argument('--entropy_cof', type=float, default=0, help='The entropy coefficient for PPO')
    p.add_argument('--mse_cof', type=float, default=1, help='The MSE coefficient for PPO')

    p.add_argument('--print_interval', type=int, default=20, help='Print interval during training')
    p.add_argument('--no_controller', default=False, action='store_true', help='Whether to use the controller')
    p.add_argument('--log_extra', default=False, action='store_true', help='Whether to log extra information')
    p.add_argument('--log_extra_interval', type=int, default=1000, help='The log interval during training')

    config = p.parse_args()

    time_stamp = time.strftime("%F-%H-%M-%S")
    log_dir = os.path.join(config.output_dir, time_stamp, 'log')
    model_dir = os.path.join(config.output_dir, time_stamp, 'model')

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
        f.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(config).items()]))

    for k in list(vars(config).keys()):
        logger.debug('%s: %s' % (k, vars(config)[k]))

    apply_seed(config.seed)
    train()
