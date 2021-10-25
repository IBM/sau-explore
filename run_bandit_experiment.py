import argparse
import os
import time

import numpy as np
import torch

import baseline_agents
import datasets
import sau_agents
from lgenrl import DCBTrainer
from utils import ddict


parser = argparse.ArgumentParser(description="Deep Contextual Bandit Experiments")

parser.add_argument("--agent", type=str, default="linear",
                    help="name of bandit algorithm (options: linear|neurallinear|lineargreedy|neuralgreedy|saulinear_sampling|saulinear_ucb|sauneural_sampling|sauneural_ucb|uniform)")
parser.add_argument("--bandit", type=str, default="mushroom",
                    help="name of bandit dataset (options: mushroom|statlog|covertype|adult|financial|jester|census|wheel)")
parser.add_argument("--delta", type=float, default=0.5,
                    help="delta parameter of the wheel bandit between 0.0 and 1.0 (it is ignored if bandit is not wheel)")
parser.add_argument("--run", type=int, default=0,
                    help="run number")
parser.add_argument("--experiment", type=int, default=None,
                    help="tells program to run an experiment (starts from 1, corresponding to a combination of bandit x agent x run)")
parser.add_argument("--use_cuda", action="store_true", default=False,
                    help="enables CUDA training")
parser.add_argument("--download", action="store_true", default=False,
                    help="download datasets if missing")

args, _ = parser.parse_known_args()
args.device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"


BANDITS = {
    "mushroom": {"name": "MushroomDataBandit", "n_samples": 50000},
    "statlog": {"name": "StatlogDataBandit", "n_samples": 43500},  # same as dataset size
    # Smaller sample than showdown paper (150k)
    "covertype": {"name": "CovertypeDataBandit", "n_samples": 50000},
    "adult": {"name": "AdultDataBandit", "n_samples": 45222},  # same as dataset size
    "financial": {"name": "FinancialDataBandit", "n_samples": 3713},  # same as dataset size
    "jester": {"name": "JesterDataBandit", "n_samples": 19181},  # same as dataset size
    # Smaller sample than showdown paper (250k)
    "census": {"name": "CensusDataBandit", "n_samples": 25000},
    "wheel": {"name": "WheelBandit", "n_samples": 2000},
}

PARAMS = {
    "mushroom": {
        "linear": {"a0": 30.0, "b0": 35.0, "lambda_prior": 20.0},  # LinFullPost-MR
        # NeuralLinear-MR
        "neurallinear": {
            "a0": 12.0,
            "b0": 30.0,
            "lambda_prior": 23.0,
        },
        "lineargreedy": {"lambda_prior": 20.0},  # LinFullPost-MR
        "neuralgreedy": {},  # RMS
        "saulinear_sampling": {"lambda_prior": 20.0},  # LinFullPost-MR
        "saulinear_ucb": {"lambda_prior": 20.0},  # LinFullPost-MR
        "sauneural_sampling": {
            "strategy": "sampling",
        },
        "sauneural_ucb": {
            "strategy": "ucb",
        },
    },
    "statlog": {
        "linear": {"a0": 35.0, "b0": 5.0, "lambda_prior": 20.0},  # LinFullPost-SL
        # NeuralLinear-SL
        "neurallinear": {
            "a0": 38.0,
            "b0": 1.0,
            "lambda_prior": 1.5,
        },
        "lineargreedy": {"lambda_prior": 20.0},  # LinFullPost-MR
        "neuralgreedy": {},  # RMS
        "saulinear_sampling": {"lambda_prior": 20.0},  # LinFullPost-MR
        "saulinear_ucb": {"lambda_prior": 20.0},  # LinFullPost-SL
        "sauneural_sampling": {
            "strategy": "sampling",
        },
        "sauneural_ucb": {
            "strategy": "ucb",
        },
    },
    "covertype": {
        "linear": {"a0": 35.0, "b0": 5.0, "lambda_prior": 20.0},  # LinFullPost-SL
        # NeuralLinear-SL
        "neurallinear": {
            "a0": 38.0,
            "b0": 1.0,
            "lambda_prior": 1.5,
        },
        "lineargreedy": {"lambda_prior": 20.0},  # LinFullPost-MR
        "neuralgreedy": {},  # RMS
        "saulinear_sampling": {"lambda_prior": 20.0},  # LinFullPost-MR
        "saulinear_ucb": {"lambda_prior": 20.0},  # LinFullPost-SL
        "sauneural_sampling": {},
        "sauneural_ucb": {},
    },
    "adult": {
        "linear": {"a0": 35.0, "b0": 5.0, "lambda_prior": 20.0},  # LinFullPost-SL
        # NeuralLinear-SL
        "neurallinear": {
            "a0": 38.0,
            "b0": 1.0,
            "lambda_prior": 1.5,
        },
        "lineargreedy": {"lambda_prior": 20.0},  # LinFullPost-MR
        "neuralgreedy": {},  # RMS
        "saulinear_sampling": {"lambda_prior": 20.0},  # LinFullPost-MR
        "saulinear_ucb": {"lambda_prior": 20.0},  # LinFullPost-SL
        "sauneural_sampling": {},
        "sauneural_ucb": {},
    },
    "financial": {
        "linear": {"a0": 6.0, "b0": 6.0, "lambda_prior": 0.25},  # LinFullPost
        # NeuralLinear-SL
        "neurallinear": {
            "a0": 38.0,
            "b0": 1.0,
            "lambda_prior": 1.5,
        },
        "lineargreedy": {"lambda_prior": 0.25},  # LinFullPost-MR
        "neuralgreedy": {},  # RMS
        "saulinear_sampling": {"lambda_prior": 0.25},  # LinFullPost-MR
        "saulinear_ucb": {"lambda_prior": 0.25},  # LinFullPost-SL
        "sauneural_sampling": {},
        "sauneural_ucb": {},
    },
    "jester": {
        "linear": {"a0": 35.0, "b0": 5.0, "lambda_prior": 20.0},  # LinFullPost-SL
        # NeuralLinear-SL
        "neurallinear": {"a0": 38.0, "b0": 1.0, "lambda_prior": 1.5, "lr_decay": 0.05},
        "lineargreedy": {"lambda_prior": 20.0},  # LinFullPost-MR
        "neuralgreedy": {"lr_decay": 0.05},  # RMS
        "saulinear_sampling": {"lambda_prior": 20.0},  # LinFullPost-MR
        "saulinear_ucb": {"lambda_prior": 20.0},  # LinFullPost-SL
        "sauneural_sampling": {"lr_decay": 0.05},
        "sauneural_ucb": {"lr_decay": 0.05},
    },
    "census": {
        "linear": {"a0": 35.0, "b0": 5.0, "lambda_prior": 20.0},  # LinFullPost-SL
        # NeuralLinear-SL
        "neurallinear": {
            "a0": 38.0,
            "b0": 1.0,
            "lambda_prior": 1.5,
        },
        "lineargreedy": {"lambda_prior": 20.0},  # LinFullPost-MR
        "neuralgreedy": {},  # RMS
        "saulinear_sampling": {"lambda_prior": 20.0},  # LinFullPost-MR
        "saulinear_ucb": {"lambda_prior": 20.0},  # LinFullPost-SL
        "sauneural_sampling": {},
        "sauneural_ucb": {},
    },
    "wheel": {
        "linear": {"a0": 30.0, "b0": 35.0, "lambda_prior": 20.0},  # LinFullPost-SL
        "neurallinear": {
            "a0": 12.0,
            "b0": 30.0,
            "lambda_prior": 23.0,
        },  # NeuralLinear-SL
        "lineargreedy": {"lambda_prior": 20.0},  # LinFullPost-MR
        "neuralgreedy": {},  # RMS
        "saulinear_sampling": {"lambda_prior": 20.0},  # LinFullPost-MR
        "saulinear_ucb": {"lambda_prior": 20.0},  # LinFullPost-SL
        "sauneural_sampling": {"init_lr": 0.01},
        "sauneural_ucb": {"init_lr": 0.01},
    },
}


# Arguments common to all models
all_kwargs = {"init_pull": 2, "device": args.device}

# Arguments common to neural models
neural_kwargs = {
    **all_kwargs,
    "dropout_p": None,
    "init_lr": 0.003,
    "lr_decay": 0.0,
    "lr_reset": False,
    "hidden_dims": [100, 100],
}

AGENTS = {
    "uniform": ["UniformAgent", {"device": args.device}],
    "linear": ["LinearUCBAgent", all_kwargs],
    "neurallinear": ["NeuralLinearUCBAgent", neural_kwargs],
    "lineargreedy": ["LinearGreedyAgent", all_kwargs],
    "neuralgreedy": ["NeuralGreedyAgent", neural_kwargs],
    "saulinear_sampling": ["SAULinearAgent", {**all_kwargs, "strategy": "sampling"}],
    "saulinear_ucb": ["SAULinearAgent", {**all_kwargs, "strategy": "ucb"}],
    "sauneural_sampling": ["SAUNeuralAgent", {**neural_kwargs, "strategy": "sampling"}],
    "sauneural_ucb": ["SAUNeuralAgent", {**neural_kwargs, "strategy": "ucb"}],
}


def load_bandit(bandit_name, download=args.download):
    DATA_PATH = os.path.expanduser("~") + "/data/" + bandit_name
    name = BANDITS[bandit_name]["name"]
    bandit = getattr(datasets, name)(download=download, path=DATA_PATH, delta=args.delta)
    return bandit


def load_agent(bandit_name, agent_name):
    agent_class, agent_params = AGENTS[agent_name]
    if hasattr(baseline_agents, agent_class):
        agent = getattr(baseline_agents, agent_class)
    elif hasattr(sau_agents, agent_class):
        agent = getattr(sau_agents, agent_class)
    bandit = load_bandit(bandit_name)
    if bandit_name in PARAMS and agent_name in PARAMS[bandit_name]:
        params = {**agent_params, **PARAMS[bandit_name][agent_name]}
    else:
        params = agent_params
    return agent(bandit, **params)


def train(bandit_name, agent_name, n_samples, logdir_label=""):
    bandit = load_bandit(bandit_name)
    agent = load_agent(bandit_name, agent_name)

    dirname = f"./logs/{bandit_name}_{agent_name}"
    if logdir_label:
        dirname = dirname + "_" + str(logdir_label)
    trainer = DCBTrainer(agent, bandit, logdir=dirname)

    # update_interval = t_f, train_epochs = t_s
    res = trainer.train(
        timesteps=n_samples, batch_size=64, update_after=20, update_interval=20, train_epochs=10
    )
    return res, trainer


def run_experiment(bandit_name, agent_name, rep=0):
    # rng seeds
    np.random.seed(rep)
    torch.manual_seed(rep)
    torch.cuda.manual_seed(rep)

    # save results
    LOG = ddict()
    LOG._save(f"output/{bandit_name}_{agent_name}_{rep}", date=False)

    exp_name = f"{bandit_name}__{agent_name}__{rep}"
    print("---------------------------------------")
    print(f"- Experiment {exp_name}:")
    print("---------------------------------------")

    start_time = time.time()
    n_samples = BANDITS[bandit_name]["n_samples"]
    res, trainer = train(bandit_name, agent_name, n_samples, logdir_label=rep)
    end_time = time.time()

    LOG[exp_name] = {"rewards": res["rewards"], "regrets": res["regrets"]}
    LOG[exp_name + "_time"] = end_time - start_time
    LOG._save()

    return LOG, trainer


if __name__ == "__main__":
    if args.experiment is None:
        LOG, trainer = run_experiment(args.bandit, args.agent, args.run)
    else:
        EXPERIMENTS = list(AGENTS.keys())
        n_exp = len(EXPERIMENTS)

        exp = (args.experiment - 1) % n_exp
        run = int((args.experiment - 1) / n_exp)

        LOG, trainer = run_experiment(args.bandit, EXPERIMENTS[exp], run)
