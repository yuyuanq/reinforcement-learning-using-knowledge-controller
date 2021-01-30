import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
from pygame.constants import K_w


class FlappyBirdEnv:
    def __init__(self, seed=0, display_screen=False):
        self.env = PLE(FlappyBird(pipe_gap=150), fps=30, display_screen=display_screen, rng=seed)
        self.step_count = 0
        self.max_step = 300

    def to_state(self, game_state_dict):
        state_list = [i for i in game_state_dict.values()]
        state_list = state_list[:5]
        state_list[3] = state_list[0] - state_list[3]
        state_list[4] = state_list[0] - state_list[4]
        return np.array(state_list) / 100

    def reset(self):
        self.env.init()
        self.step_count = 0
        return self.to_state(self.env.getGameState())

    def step(self, a):
        if a == 1:
            a = K_w
        else:
            a = None

        r = self.env.act(a)

        if r == -5:
            r = -10
        else:
            r = 0.1

        done = self.env.game_over()
        self.step_count += 1
        if self.step_count >= self.max_step:
            done = True

        s_prime = self.to_state(self.env.getGameState())
        return s_prime, r, done, {}

    @staticmethod
    def get_space_dim():
        return 5, 2
