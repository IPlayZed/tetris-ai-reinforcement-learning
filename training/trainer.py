import random
import time
from math import log

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from tetris_gym.envs.tetris_gym import TetrisGym
from tetris_gym.utils.eval_utils import evaluate, create_videos
from tetris_gym.wrappers.observation import ExtendedObservationWrapper


class Trainer:
    DEFAULT_LEARNING_TIMESTAMPS: int = 1_000_000_000
    DEFAULT_NUMBER_OF_ENVIRONMENTS: int = 4

    def __init__(self, environment: VecEnv = None, env_n: int = DEFAULT_NUMBER_OF_ENVIRONMENTS,
                 policy: str = "MultiInputPolicy", verbosity: int = 0,
                 default_reward: bool = True, save_path_base: str = "models",
                 log_path_base: str = "logs", evaluation_environment: gym.Env = None,
                 evaluation_videos_path_base: str = "videos", model_path: str = None,
                 lr_scheduler=None):
        if environment is None:
            self._environment: VecEnv = Trainer.get_vectorized_gym_env(env_n=env_n, default_reward=default_reward)
        else:
            self._environment: VecEnv = environment
        self._policy: str = policy
        self._verbosity: int = verbosity
        self._save_path_base: str = save_path_base
        self._log_path_base: str = log_path_base
        self._env_n: int = env_n
        if evaluation_environment is None:
            self._evaluation_environment = Trainer.get_wrapped_env(default_reward=default_reward)
        else:
            self._evaluation_env: gym.Env = evaluation_environment
        self._evaluation_videos_path_base: str = evaluation_videos_path_base
        self._tensorboard_log: str = f"{log_path_base}/tensorboard"
        if lr_scheduler is None:
            self._lr_scheduler = Trainer.lr_schedule_exp_decay()
        else:
            self._lr_scheduler = lr_scheduler

        # Load existing model or create a new one
        if model_path is None:
            self._model: PPO = PPO(policy=self._policy, verbose=self._verbosity, env=self._environment,
                                   learning_rate=self._lr_scheduler)
        else:
            self._model: PPO = PPO.load(path=model_path, verbosity=self._verbosity, env=self._environment)
            self._model.learning_rate = self._lr_scheduler

        self._model.tensorboard_log = self._tensorboard_log
        print(f"Using network:\n{self._model.policy}")

    @staticmethod
    def lr_schedule_exp_decay(initial_value: float = 5e-4, rate: float = 1.01):
        """
        Learning rate schedule:
            Exponential decay by factors of 2

        :param initial_value: Initial learning rate.
        :param rate: Exponential rate of decay. High values mean fast early drop in LR
        :return: schedule that computes
          current learning rate depending on remaining progress
        """

        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            if progress_remaining <= 0:
                return 3e-4

            return initial_value * 2 ** (rate * log(progress_remaining))

        return func

    @staticmethod
    def get_wrapped_env(default_reward: bool = True) -> ExtendedObservationWrapper:
        return ExtendedObservationWrapper(TetrisGym(), default_reward=default_reward)

    @staticmethod
    def get_vectorized_gym_env(env_n: int = DEFAULT_NUMBER_OF_ENVIRONMENTS, default_reward: bool = True) -> VecEnv:
        return make_vec_env(
            lambda: Trainer.get_wrapped_env(default_reward=default_reward),
            n_envs=env_n
        )

    def train(self, learning_timestamps: int = DEFAULT_LEARNING_TIMESTAMPS, show_progress_bar=True,
              save_frequency: int = 1_000_000, save_name_prefix: str = "autosave",
              save_name_suffix: str = "", tensorboard_logname: str = None, save_loglevel: int = 2,
              callback: BaseCallback = None):
        if callback is None:
            learning_callback = CheckpointCallback(
                save_freq=max(save_frequency // self._env_n, 1),
                save_path=f"{self._save_path_base}/autosave",
                name_prefix=save_name_prefix,
                verbose=save_loglevel
            )
        else:
            learning_callback = callback
        if tensorboard_logname is None:
            self._model.learn(total_timesteps=learning_timestamps, progress_bar=show_progress_bar,
                              callback=learning_callback)
        else:
            self._model.learn(total_timesteps=learning_timestamps, progress_bar=show_progress_bar,
                              tb_log_name=tensorboard_logname, callback=learning_callback)

        print("Training done, saving final model.")

        self._model \
            .save(f"{self._save_path_base}/completed/{learning_timestamps}/{save_name_suffix}")

    def evaluate(self, evaluation_episodes: int = 100_000, seed: int = None, save_videos: bool = True,
                 video_episodes: int = 10):
        print("Starting evaluation...")
        if seed is None:
            random.seed(time.time())
            self._model.seed = random.randint(0, 100_000)
        print("Evaluation score: {}".format(evaluate(self._evaluation_env, self._model, evaluation_episodes)))
        if save_videos:
            print("Creating videos...")
            create_videos(env=self._evaluation_env, model=self._model, ep_num=video_episodes)
            self._evaluation_env.reset()
