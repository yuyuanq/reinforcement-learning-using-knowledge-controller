import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
from pygame.constants import K_w


class FlappyBirdEnv:
    def __init__(self, seed=0, display_screen=False, max_episode_length=2000, pipe_gap=150, delay_step=1):
        game = FlappyBird(pipe_gap=pipe_gap)
        self.p = PLE(game, fps=30, display_screen=display_screen, rng=seed)
        self.max_l = max_episode_length

        self.step_count = 0
        self.delay_step = delay_step
        self.pass_pipe = 0

    def to_state(self, game_state_dict):
        state_list = [i for i in game_state_dict.values()]
        state_list = state_list[:5]
        state_list[3] = state_list[0] - state_list[3]
        state_list[4] = state_list[0] - state_list[4]
        return np.array(state_list)

    def reset(self):
        self.p.init()
        self.p.previous_score = 0
        self.step_count = 0
        self.pass_pipe = 0
        return self.to_state(self.p.getGameState())

    def step(self, act):
        if act == 0:
            act = None
        else:
            act = 119

        r = self.p.act(act)
        done = self.p.game_over()
        snext = self.to_state(self.p.getGameState())

        if r == 1:
            self.pass_pipe += 1

        if self.pass_pipe == self.delay_step:
            reward = self.delay_step
            self.pass_pipe = 0
        else:
            reward = 0
        if done:
            reward += -5
        info = {}
        self.step_count += 1
        if self.step_count >= self.max_l:
            done = True
        return snext, reward, done, info

    @staticmethod
    def get_space_dim():
        return 5, 2
