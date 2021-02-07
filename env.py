import gym


class GymEnvironment:
    def __init__(self, env_name, delay_step=1, seed=0, max_steps=1500):
        assert delay_step > 0

        self.env = gym.make(env_name)
        self.env.seed(seed)

        self.max_steps = max_steps
        self.delay_step = delay_step
        self.delay_counter = 0
        self.delay_reward = 0
        self.steps = 0

    def reset(self):
        s = self.env.reset()
        self.delay_counter = 0
        self.steps = 0
        return s

    def render(self):
        self.env.render()

    def step(self, act):
        s, r, d, info = self.env.step(act)

        self.steps += 1
        self.delay_counter += 1
        self.delay_reward += r

        if self.steps == self.max_steps:
            d = True

        if self.delay_counter == self.delay_step:
            r = self.delay_reward
            self.delay_reward = 0
            self.delay_counter = 0

        return s, r, d, info

    def get_space_dim(self):
        try:
            return self.env.observation_space.shape[0], self.env.action_space.n
        except AttributeError:
            return self.env.observation_space.shape[0], self.env.action_space.shape[0]

    def close(self):
        self.env.close()
