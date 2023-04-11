import gym
from stable_baselines3 import PPO
from tetris_gym.wrappers.observation import ExtendedObservationWrapper


class Agent:
    """
    This agent was created for evaluating the
    """

    def __init__(self, env: gym.Env) -> None:
        self._model_path: str = "agent/fml.zip"
        self._model = PPO.load(self._model_path)
        self._wrapper: gym.Wrapper = ExtendedObservationWrapper(env)

    def act(self, observation):
        """
        Based on the observation, it returns the next step. This function will give the agent's function.

        Args:
            observation: The observation from the environment.

        Returns:
            The model's action and the next hidden state (used in recurrent policies).
        """

        # If we modified the observations when teaching, we must also provide that modification at the time of
        # evaluation.
        wrapped_observation = self._wrapper.observation(observation)

        return self._model.predict(wrapped_observation, deterministic=True)


