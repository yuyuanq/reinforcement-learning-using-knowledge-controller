import numpy as np
import configargparse
import torch
from torch.distributions import Categorical, MultivariateNormal, Normal
from tensorboardX import SummaryWriter
from env import GymEnvironment
from agent import PPO
import os
from logger import logger
from env_flappybird_kogun import FlappyBirdEnv
import time
import wandb


# Single CPU
cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def apply_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    wandb.init(
        project='knowledge-rl',
        name=f'seed_{config.seed}',
        group=f'{config.env}_{config.no_controller}_{config.info}',
        dir='./output',
    )

    render = False
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
        for _ in range(config.t_horizon ** 10):  # TODO
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
                wandb.log({'ep_reward': ep_reward, 'ep_count': ep_count, 'steps': steps, 'update_count_eq': steps // config.t_horizon})
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
                    "episode: {}, update count: {}, reward: {:.1f}, steps:{}, p_cof: {:.2f}"
                    .format(ep_count, update_count, last_ep_reward, steps,
                            model.actor.p_cof))

        if update_count % config.save_interval == 0:
            torch.save(
                model.state_dict(),
                os.path.join(model_dir, 'model_{}.pkl'.format(update_count)))

        writer.add_scalar('reward/update_reward', last_ep_reward, update_count)

        if update_count_eq >= config.max_update:  # TODO
            break

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
                   default=0.9,
                   help='The coefficient of the start of p')
    p.add_argument('--p_cof_end',
                   type=float,
                   default=0.1,
                   help='The coefficient of the end of p')
    p.add_argument('--p_total_step',
                   type=float,
                   default=1000,
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
    p.add_argument('--info',
                   type=str,
                   default='')

    config = p.parse_args()

    time_stamp = time.strftime("%F-%H-%M-%S")
    log_dir = os.path.join(config.output_dir, config.env,
                           str(config.no_controller), time_stamp, 'log')
    model_dir = os.path.join(config.output_dir, config.env,
                             str(config.no_controller), time_stamp, 'model')

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
        f.write('\n'.join(
            ["%s: %s" % (key, value) for key, value in vars(config).items()]))

    for k in list(vars(config).keys()):
        logger.debug('%s: %s' % (k, vars(config)[k]))

    apply_seed(config.seed)
    train()
