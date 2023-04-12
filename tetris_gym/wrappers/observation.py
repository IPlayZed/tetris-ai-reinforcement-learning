import gym
from gym import spaces

import numpy as np

from tetris_gym.utils.board_utils import get_heights, get_bumps_from_heights


class ExtendedObservationWrapper(gym.ObservationWrapper, gym.RewardWrapper):

    def __init__(self, env, default_reward: bool = True):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            "board": env.observation_space["board"],
            "piece": env.observation_space["piece"],
            "heights": spaces.Box(
                low=0,
                high=env.height,
                shape=(env.width,),
                dtype=int,
            ),
            "bumps": spaces.Box(
                low=0,
                high=env.height,
                shape=(env.width - 1,),
                dtype=int,
            )
        })
        self._default_reward: bool = default_reward

    def reward(self, reward):

        final_reward: float = 0

        final_reward += (self.env.check_cleared_rows(self.env.board)[0] ** 4) * self.env.width

        if self.env.gameover:
            final_reward -= 1
        else:
            final_reward += 1

        if not self._default_reward:
            final_reward += self.custom_reward()  # We add it as this function is return a penalty

        return final_reward

    def custom_reward(self):
        final_penalty: float = 0
        obs = self.observation(self.env.get_observations())
        heights = obs["heights"]
        bumps = obs["bumps"]

        final_penalty -= np.max(heights) / 500  # Penalty for heights
        final_penalty -= np.max(np.abs(bumps)) / 500  # Penalty for bumps
        final_penalty -= self.env.get_holes(obs["board"]) / 500  # Penalty for holes

        return final_penalty

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        custom_reward = self.reward(reward)

        # Return the modified step results
        return self.observation(obs), custom_reward, done, info

    def observation(self, obs):
        board = obs["board"]
        piece = obs["piece"]

        heights = get_heights(board)

        bumps = get_bumps_from_heights(heights)

        obs = {
            "board": board,
            "piece": piece,
            "heights": heights,
            "bumps": bumps
        }

        return obs

    @staticmethod
    def normalize(number, max_value: int | float, min_value: int | float = 0):
        if max_value - min_value == 0:
            return 0
        return (number - min_value) / (max_value - min_value)
