import gym
from gym import spaces

import numpy as np

from tetris_gym.utils.board_utils import get_heights, get_bumps_from_heights


def normalize(number, max_value: int | float, min_value: int | float = 0):
    if max_value - min_value == 0:
        return 0
    return (number - min_value) / (max_value - min_value)


class ExtendedObservationWrapper(gym.ObservationWrapper, gym.RewardWrapper):

    def __init__(self, env):
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

    def reward(self, reward):
        # Implement your custom reward calculation here
        # For example, you can use the 'heights' and 'bumps' information from the observation
        # to calculate a reward based on the current state of the board

        obs = self.observation(self.env.get_observations())
        heights = obs["heights"]
        bumps = obs["bumps"]

        reward_cleared_rows = (self.env.check_cleared_rows(self.env.board)[0] ** 4) * self.env.width

        penalty_heights = np.max(heights) / 500

        penalty_bumps = np.max(np.abs(bumps)) / 500

        if self.env.gameover:
            is_game_over = -1
        else:
            is_game_over = 1

        penalty_holes = self.env.get_holes(obs["board"]) / 500

        return reward_cleared_rows + is_game_over# - penalty_heights - penalty_holes - penalty_holes

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


class HeightsObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            "heights": env.observation_space["heights"],
            "bumps": env.observation_space["bumps"],
            "piece": env.observation_space["piece"]
        })

    def observation(self, obs):
        return {
            "heights": obs["heights"],
            "bumps": obs["bumps"],
            "piece": obs["piece"]
        }
