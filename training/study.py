import random
import time
from typing import Any, Optional
from typing import Dict

import gym
import optuna
import torch
import torch.nn as nn
from optuna import Study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv

from tetris_gym.envs.tetris_gym import TetrisGym
from tetris_gym.utils.eval_utils import evaluate, create_videos
from tetris_gym.wrappers.observation import ExtendedObservationWrapper

best_model: Optional[PPO] = None
model: Optional[PPO] = None
n_best_found: int = 0

N_TRIALS: int = 500
N_STARTUP_TRIALS: int = 50
N_EVALUATIONS: int = 1000
N_TIMESTAMPS: int = 1_000_000


def calculate_eval_freq(n_timestamps=N_TIMESTAMPS, n_evaluations=N_EVALUATIONS) -> int:
    return int(n_timestamps // n_evaluations)


EVAL_FREQ: int = calculate_eval_freq()
N_EVAL_EPISODES: int = 5

DEFAULT_HYPERPARAMS: dict[str, int, Optional[VecEnv]] = {
    "policy": "MultiInputPolicy",
    "verbose": 0,
    "env": VecEnv,
}


def get_gym_env() -> TetrisGym:
    return ExtendedObservationWrapper(TetrisGym())


def get_vectorized_gym_env(n_envs: int = 4) -> VecEnv:
    return make_vec_env(
        lambda: get_gym_env(),
        n_envs=n_envs
    )


def reset_optuna_params_to_defaults() -> None:
    """ Call this function if you wish to create multiple models with different study parameters after creating a
    model"""
    global N_TRIALS, N_STARTUP_TRIALS, N_EVALUATIONS, N_TIMESTAMPS, EVAL_FREQ, N_EVAL_EPISODES
    N_TRIALS = 500
    N_STARTUP_TRIALS = 25
    N_EVALUATIONS = 1000
    N_TIMESTAMPS = 100_000
    EVAL_FREQ = calculate_eval_freq()
    N_EVAL_EPISODES = 5


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyper-parameters."""

    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    learning_rate = trial.suggest_float("lr", 1e-5, 0.1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    # net_arch = trial.suggest_categorical("net_arch", ["FQN_tiny", "FQN_small", "FQN_medium", "FQN_large"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu", "leaky", "selu"])

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    '''
    match net_arch:
            case "FQN_tiny":
                net_arch = {"pi": [32], "vf": [32]}
            case "FQN_small":
                net_arch = {"pi": [32, 32], "vf": [32, 32]}
            case "FQN_medium":
                net_arch = {"pi": [64, 64], "vf": [64, 64]}
            case "FQN_large":
                net_arch = {"pi": [128, 128], "vf": [128, 128]}
    '''

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky": nn.LeakyReLU,
        "selu": nn.SELU
    }[activation_fn]

    random.seed(time.time())
    current_rand = random.randint(0, 100_000)
    print(f"Using seed {current_rand} for model.")

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "seed": current_rand,
        "policy_kwargs": {
            #"net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        }
    }


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
            self,
            eval_env: gym.Env,
            trial: optuna.Trial,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def update_best_model(study: Study, trial: FrozenTrial) -> None:
    """ Updates the best model if a study exceeded it."""
    global best_model, n_best_found
    if study.best_trial == trial:
        n_best_found += 1
        print(f"Found better model. This the {n_best_found}. time.")
        best_model = model


def objective(trial: optuna.Trial) -> float:
    """Then function callback describing the study objective."""
    global model

    # noinspection PyTypeChecker
    # reason: the type checker identifies the indexed element as string (?)
    DEFAULT_HYPERPARAMS["env"] = get_vectorized_gym_env()
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyper-parameters.
    kwargs.update(sample_ppo_params(trial))

    model = PPO(**kwargs)
    model.tensorboard_log = "log/objective/"

    # Create env used for evaluation. I use the default provided.
    eval_env: Monitor = Monitor(get_gym_env())
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback: TrialEvalCallback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    print("Will try to learn with parameters: {}".format(kwargs))
    try:
        model.learn(N_TIMESTAMPS, callback=eval_callback, progress_bar=True, tb_log_name="learning")
    except AssertionError as error:
        # Sometimes, random hyperparams can generate NaN.
        print(error)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


def create_model(n_trials: int = N_TRIALS,
                 n_startup_trials: int = N_STARTUP_TRIALS,
                 n_evaluations: int = N_EVALUATIONS,
                 n_timestamps: int = N_TIMESTAMPS,
                 n_eval_episodes: int = N_EVAL_EPISODES) -> None:
    """ Runs a study with hyperparameter search and saves the model."""

    global N_TRIALS, N_STARTUP_TRIALS, N_EVALUATIONS, N_TIMESTAMPS, N_EVAL_EPISODES, EVAL_FREQ

    N_TRIALS = n_trials
    N_STARTUP_TRIALS = n_startup_trials
    N_EVALUATIONS = n_evaluations
    N_TIMESTAMPS = n_timestamps
    N_EVAL_EPISODES = n_eval_episodes

    EVAL_FREQ = calculate_eval_freq()

    sampler: TPESampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner: MedianPruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study: Study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[update_best_model])
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    best_trial: FrozenTrial = study.best_trial
    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in best_trial.user_attrs.items():
        print("    {}: {}".format(key, value))

    path: str = f"best_model/n_timestamps_{n_timestamps}/n_trials_{n_trials}"

    best_model.save(f"models/{path}")

    create_videos(get_gym_env(), best_model, ep_num=10, folder=f"videos/{path}")
    score: int = evaluate(
        get_gym_env(),
        best_model,
        ep_num=1000
    )
    print("Score: {}".format(score))


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    create_model()

    reset_optuna_params_to_defaults()
