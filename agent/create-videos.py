from stable_baselines3 import PPO

from tetris_gym.envs.tetris_gym import TetrisGym
from tetris_gym.utils.eval_utils import create_videos
from tetris_gym.wrappers.observation import ExtendedObservationWrapper

env = ExtendedObservationWrapper(TetrisGym(width=10, height=20, seed=42), False)
create_videos(env, PPO.load("autosave_145000000_steps.zip"), ep_num=10)
