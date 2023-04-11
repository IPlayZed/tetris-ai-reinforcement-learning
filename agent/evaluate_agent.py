from stable_baselines3 import PPO

from agent.agent import Agent
from tetris_gym.envs.tetris_gym import TetrisGym
from tetris_gym.utils.eval_utils import evaluate_agent, create_videos

# Környezet létrehozása
env = TetrisGym(width=10, height=20, seed=812468)


agent = Agent(env)

print(evaluate_agent(env, agent, 100))

