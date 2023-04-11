import gym
from gym import spaces
import numpy as np

from tetris_gym.envs.tetris import Tetris
from tetris_gym.utils.board_utils import binarize_board

class TetrisGym(Tetris, gym.Env):
    
    def __init__(self,
                 height=20,
                 width=10,
                 pieces=["O", "I", "T", "S", "Z", "L", "J"],
                 block_size=20,
                 max_steps=1000,
                 seed=42):
        super().__init__(height, width, pieces, block_size, seed)

        self.max_steps = max_steps

        self.step_counter = 0
    
    def reset(self):
        super().reset()

        self.step_counter = 0

        return self.get_observations()

    def step(self, action):
        column, rotation = action

        column = int(column)
        rotation = int(rotation)

        rotated_piece = self.piece
        for _ in range(rotation):
            rotated_piece = self.rotate(rotated_piece)

        piece_width = len(rotated_piece[0])

        if piece_width + column > self.width:
            column = self.width - piece_width

        res = super().step((column, rotation), False, None)

        self.step_counter += 1

        done = res[1]

        if self.step_counter >= self.max_steps:
          done = True

        return self.get_observations(), res[0], done, {}
    
    def get_observations(self):
        bin_board = binarize_board(self.board)

        return {
            "board": bin_board,
            "piece": max(max(self.piece)) - 1,
        }

    @property
    def observation_space(self):
        """Override the superclass property.
        :return: Box obs space.
        """
        
        return spaces.Dict({
            "board": spaces.Box(
                low=0,
                high=1,
                shape=(self.height, self.width),
                dtype=np.uint8,
            ),
            "piece": spaces.Discrete(7),

        })

    @property
    def action_space(self):
        return spaces.MultiDiscrete([self.width, 4])

    def get_next_states(self):
        states = super().get_next_states()

        for r in range(4):
          r_copy = 0
          if (0, r) not in states:
              
              r_copy = -1 # in case of O piece

              if (0, r - 2) in states:
                r_copy = -2
          
          for a in range(10):
              if (a,r) not in states:
                  if r_copy == 0:
                    states[(a, r)] = states[(a - 1, r)]
                  else:
                    states[(a, r)] = states[(a, r + r_copy)]

        return states