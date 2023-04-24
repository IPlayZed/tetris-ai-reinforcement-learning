import gym
from stable_baselines3 import PPO

from agent.observation import ExtendedObservationWrapper
from tetris_gym.envs.tetris_gym import TetrisGym


def get_eval_env(seed=42) -> TetrisGym:
    return TetrisGym(width=10, height=20, seed=seed)


class Agent:
    """
    This agent was created for evaluating the
    """

    def __init__(self, evaluation_environment: gym.Env = get_eval_env(),
                 model_load_path: str = "model.zip") -> None:
        self._model_path: str = model_load_path
        self._model = PPO.load(self._model_path)
        self._wrapper: gym.Wrapper = ExtendedObservationWrapper(evaluation_environment, False)

    def act(self, observation):
        """
        Based on the observation, it returns the next step. This function will give the agent's function.

        Args:
            observation: The observation from the environment.

        Returns:
            The model's action and the next hidden state (used in recurrent policies).
        """

        return self._model.predict(self._wrapper.observation(observation), deterministic=True)
