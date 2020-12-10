import gym


class Environment:
    def __init__(self, env_name, delay_step=1, seed=0):
        self.env = gym.make(env_name)
        self.env.seed(seed)
        assert delay_step > 0, 'Delay step should be greater than 0'
        self.delay_step = delay_step
        self.counter = 0

    def reset(self):
        s = self.env.reset()
        self.counter = 0
        return s

    def step(self, act):
        s, r, d, info = self.env.step(act)
        self.counter += 1
        if self.counter == self.delay_step:
            r = self.delay_step
            self.counter = 0
        else:
            r = 0
        return s, r, d, info

    def get_space_dim(self):
        return self.env.observation_space.shape[0], self.env.action_space.n

    def close(self):
        self.env.close()

