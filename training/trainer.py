import time
import random

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from tetris_gym.envs.tetris_gym import TetrisGym
from tetris_gym.utils.eval_utils import evaluate, create_videos
from tetris_gym.wrappers.observation import ExtendedObservationWrapper
from training.study import get_vectorized_gym_env


class Trainer:
    DEFAULT_LEARNING_TIMESTAMPS: int = 100_000_000
    DEFAULT_NUMBER_OF_ENVIRONMENTS: int = 4

    def __init__(self, environment: gym.Env = get_vectorized_gym_env(DEFAULT_NUMBER_OF_ENVIRONMENTS),
                 policy: str = "MultiInputPolicy", verbosity: int = 0, save_path_base: str = "models",
                 log_path_base: str = "logs",
                 evaluation_environment: gym.Env = ExtendedObservationWrapper(TetrisGym()),
                 evaluation_videos_path_base: str = "videos"):
        self._environment = environment
        self._policy = policy
        self._verbosity = verbosity
        self._save_path_base = save_path_base
        self._log_path_base = log_path_base
        self._evaluation_env = evaluation_environment
        self._evaluation_videos_path_base = evaluation_videos_path_base

        self._model = PPO(policy=self._policy, verbose=self._verbosity, env=self._environment,
                          tensorboard_log=f"{log_path_base}/tensorboard")

    @staticmethod
    def get_vectorized_gym_env(self: int = DEFAULT_NUMBER_OF_ENVIRONMENTS) -> VecEnv:
        return make_vec_env(
            lambda: ExtendedObservationWrapper(TetrisGym()),
            n_envs=self
        )

    def train(self, learning_timestamps: int = DEFAULT_LEARNING_TIMESTAMPS, show_progress_bar=True,
              save_frequency: int = 100, save_name_prefix: str = "autosave",
              save_name_suffix: str = ""):
        self._model.learn(total_timesteps=learning_timestamps, progress_bar=show_progress_bar,
                          callback=CheckpointCallback(
                              save_freq=save_frequency,
                              save_path=f"{self._save_path_base}/autosave",
                              name_prefix=save_name_prefix
                          ))

        self._model \
            .save(f"{self._save_path_base}/completed/{save_name_prefix}/{learning_timestamps}/{save_name_suffix}")

    def evaluate(self, evaluation_episodes: int = 10_000, seed: int = None, save_videos: bool = True,
                 video_episodes: int = 10):
        self._evaluation_env.reset()
        if seed is None:
            random.seed(time.time())
            self._model.seed = random.randint(0, 100_000)
        print("Evaluation score: {}".format(evaluate(self._evaluation_env, self._model, evaluation_episodes)))
        self._evaluation_env.reset()
        if save_videos:
            create_videos(env=self._evaluation_env, model=self._model, ep_num=video_episodes)
            self._evaluation_env.reset()
