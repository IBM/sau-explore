import os
from functools import partial
from multiprocessing import Process

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

from lgenrl import DCBTrainer
from run_bandit_experiment import load_agent, load_bandit

BANDIT = "mushroom"
MAX_STEPS = 1000
AGENTS = ["sauneural_ucb", "neurallinear", "neuralgreedy"]


def prepare_tasks():
    trainers = []
    for agent_name in AGENTS:
        bandit = load_bandit(BANDIT, download=True)
        agent = load_agent(BANDIT, agent_name)
        dir_name = f"./logs/{BANDIT}_{agent_name}"
        trainers.append(DCBTrainer(agent, bandit, logdir=dir_name))

    train_funcs = [
        partial(
            tr.train,
            timesteps=MAX_STEPS,
            batch_size=64,
            update_after=20,
            update_interval=20,
            train_epochs=10,
        )
        for tr in trainers
    ]
    return trainers, train_funcs


def get_animate(fig, ax, data_files, labels):
    def animate(i):
        data = [
            pd.read_csv(
                data_file,
                sep="\s+",
                header=None,
                names=[
                    "timestep",
                    "regret",
                    "reward",
                    "cumulative_regret",
                    "cumulative_reward",
                    "regret_moving_avg",
                    "reward_moving_avg",
                ],
            )
            for data_file in data_files
            if os.path.exists(data_file)
        ]

        ax.clear()
        for d, lab in zip(data, labels):
            if lab[:3] == "sau":
                linewidth = 3.0
            else:
                linewidth = 2.0
            ax.plot(
                d["timestep"],
                d["reward_moving_avg"],
                linewidth=linewidth,
                marker="o",
                markersize=2 * linewidth,
                label=lab,
            )

        ax.set_xlabel("timestep")
        ax.set_ylabel("average reward")
        ax.set_title(BANDIT + " bandit")
        ax.legend(loc="lower right")
        ax.set_xlim([0, MAX_STEPS])
        return (ax,)

    def init_func():
        for data_file in data_files:
            if os.path.exists(data_file):
                os.remove(data_file)

    return animate, init_func


def run_tasks_in_parallel(tasks, anim_func=None):
    running_tasks = [Process(target=task) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    if anim_func is not None:
        anim = anim_func()
    for running_task in running_tasks:
        running_task.join()
    if anim_func is not None:
        return anim
    else:
        return None


if __name__ == "__main__":
    trainers, train_funcs = prepare_tasks()

    fig, ax = plt.subplots()

    data_files = [os.path.join(tr.logger._logdir, "train.log") for tr in trainers]
    animate, init_func = get_animate(fig, ax, data_files, AGENTS)

    def anim_func():
        anim = FuncAnimation(fig, animate, init_func=init_func, interval=500)
        plt.show()
        return anim

    anim = run_tasks_in_parallel(train_funcs, anim_func)
