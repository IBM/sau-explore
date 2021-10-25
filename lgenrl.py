"""
This is a lightweight version of GenRL based on code from the SforAiDl/GenRL library
at https://github.com/SforAiDl/genrl and distributed under the following license:

MIT License

Copyright (c) 2020 Srivatsan Krishnan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.bandits import TransitionDB
from utils import Logger


class RMSNeuralModel(nn.Module):
    """Neural Network used in Deep Contextual Bandit Models.

    Args:
        context_dim (int): Length of context vector.
        hidden_dims (List[int], optional): Dimensions of hidden layers of network.
        n_actions (int): Number of actions that can be selected. Taken as length
            of output vector for network to predict.
        init_lr (float, optional): Initial learning rate.
        max_grad_norm (float, optional): Maximum norm of gradients for gradient clipping.
        lr_decay (float, optional): Decay rate for learning rate.
        lr_reset (bool, optional): Whether to reset learning rate ever train interval.
            Defaults to False.
        dropout_p (Optional[float], optional): Probability for dropout. Defaults to None
            which implies dropout is not to be used.
        noise_std (float): Standard deviation of noise used in the network. Defaults to 0.1

    Attributes:
        use_dropout (int): Indicated whether or not dropout should be used in forward pass.
    """
    def __init__(self, **kwargs):
        super(RMSNeuralModel, self).__init__()

        self.context_dim = kwargs.get("context_dim")
        self.hidden_dims = kwargs.get("hidden_dims")
        self.n_actions = kwargs.get("n_actions")

        self.lr_reset = kwargs.get("lr_reset", False)
        self.init_lr = kwargs.get("init_lr", 3e-4)
        self.lr_decay = kwargs.get("lr_decay", None)
        self.dropout_p = kwargs.get("dropout_p", None)
        self.use_dropout = True if self.dropout_p is not None else False
        self.max_grad_norm = kwargs.get("max_grad_norm")

        t_hidden_dims = [self.context_dim, *self.hidden_dims, self.n_actions]
        self.layers = nn.ModuleList([])
        for i in range(len(t_hidden_dims) - 1):
            self.layers.append(
                nn.Linear(t_hidden_dims[i], t_hidden_dims[i + 1]))
        self.optimizer = torch.optim.RMSprop(self.parameters(),
                                             lr=self.init_lr)
        self.lr_scheduler = (torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda i: 1 /
            (1 + self.lr_decay * i)) if self.lr_decay is not None else None)

    def forward(self, context: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Computes forward pass through the network.

        Args:
            context (torch.Tensor): The context vector to perform forward pass on.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of outputs
        """
        x = context
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            if self.dropout_p is not None and self.use_dropout is True:
                x = F.dropout(x, self.dropout_p)

        pred_rewards = self.layers[-1](x)
        return dict(x=x, pred_rewards=pred_rewards)

    def train_model(self, db: TransitionDB, epochs: int, batch_size: int):
        """Trains the network on a given database for given epochs and batch_size.

        Args:
            db (TransitionDB): The database of transitions to train on.
            epochs (int): Number of gradient steps to take.
            batch_size (int): The size of each batch to perform gradient descent on.
        """
        self.use_dropout = True if self.dropout_p is not None else False

        if self.lr_decay is not None and self.lr_reset is True:
            self._reset_lr(self.init_lr)

        for _ in range(epochs):
            x, a, y = db.get_data(batch_size)
            action_mask = F.one_hot(a, num_classes=self.n_actions)
            reward_vec = y.view(-1).repeat(self.n_actions, 1).T * action_mask
            loss = self._compute_loss(db, x, action_mask, reward_vec,
                                      batch_size)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.lr_decay is not None:
                self.lr_scheduler.step()

    def _compute_loss(self, db: TransitionDB, x: torch.Tensor,
                      action_mask: torch.Tensor, reward_vec: torch.Tensor,
                      batch_size: int) -> torch.Tensor:
        """Computes loss for the model

        Args:
            db (TransitionDB): The database of transitions to train on.
            x (torch.Tensor): Context.
            action_mask (torch.Tensor): Mask of actions taken.
            reward_vec (torch.Tensor): Reward vector recieved.
            batch_size (int): The size of each batch to perform gradient descent on.

        Returns:
            torch.Tensor: The computed loss.
        """
        results = self.forward(x)
        loss = (torch.sum(action_mask *
                          (reward_vec - results["pred_rewards"])**2) /
                batch_size)
        return loss

    def _reset_lr(self, lr: float) -> None:
        """Resets learning rate of optimizer.

        Args:
            lr (float): New value of learning rate
        """
        for o in self.optimizer.param_groups:
            o["lr"] = lr
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda i: 1 / (1 + self.lr_decay * i))


class Bandit(ABC):
    """Abstract Base class for bandits"""
    @abstractmethod
    def step(self, action: int) -> Tuple[torch.Tensor, int]:
        """Generate reward for given action and select next context.
        Args:
            action (int): Selected action.
        Returns:
            Tuple[torch.Tensor, int]: Tuple of the next context and the
                reward generated for given action
        """

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Reset bandit.
        Returns:
            torch.Tensor: Current context selected by bandit.
        """


class BanditAgent(ABC):
    """Abstract Base class for bandit solving agents"""
    @abstractmethod
    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on given context
        Args:
            context (torch.Tensor): The context vector to select action for
        Returns:
            int: The action to take
        """


class DCBAgent(BanditAgent):
    """Base class for deep contextual bandit solving agents
    Args:
        bandit (gennav.deep.bandit.data_bandits.DataBasedBandit): The bandit to solve
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".
    Attributes:
        bandit (gennav.deep.bandit.data_bandits.DataBasedBandit): The bandit to solve
        device (torch.device): Device to use for tensor operations.
    """
    def __init__(self, bandit: Bandit, device: str = "cpu", **kwargs):

        super(DCBAgent, self).__init__()

        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self._bandit = bandit
        self.context_dim = self._bandit.context_dim
        self.n_actions = self._bandit.n_actions
        self._action_hist = []
        self.init_pulls = kwargs.get("init_pulls", 3)

    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on given context
        Args:
            context (torch.Tensor): The context vector to select action for
        Note:
            This method needs to be implemented in the specific agent.
        Returns:
            int: The action to take
        """
        raise NotImplementedError

    def update_parameters(
        self,
        action: Optional[int] = None,
        batch_size: Optional[int] = None,
        train_epochs: Optional[int] = None,
    ) -> None:
        """Update parameters of the agent.
        Args:
            action (Optional[int], optional): Action to update the parameters for. Defaults to None.
            batch_size (Optional[int], optional): Size of batch to update parameters with. Defaults to None.
            train_epochs (Optional[int], optional): Epochs to train neural network for. Defaults to None.
        Note:
            This method needs to be implemented in the specific agent.
        """
        raise NotImplementedError


class BanditTrainer(ABC):
    """Bandit Trainer Class
    Args:
        agent (genrl.deep.bandit.dcb_agents.DCBAgent): Agent to train.
        bandit (genrl.deep.bandit.data_bandits.DataBasedBandit): Bandit to train agent on.
        logdir (str): Path to directory to store logs in.
        log_mode (List[str]): List of modes for logging.
    """
    def __init__(
        self,
        agent: Any,
        bandit: Any,
        logdir: str = "./logs",
        log_mode: List[str] = ["stdout"],
    ):
        self.agent = agent
        self.bandit = bandit
        self.logdir = logdir
        self.log_mode = log_mode
        self.logger = Logger(logdir=logdir, formats=[*log_mode])

    @abstractmethod
    def train(self) -> None:
        """
        To be defined in inherited classes
        """


class DCBTrainer(BanditTrainer):
    def __init__(
        self,
        agent: Any,
        bandit: Any,
        logdir: str = "./logs",
        log_mode: List[str] = ["stdout"],
    ):
        super(DCBTrainer, self).__init__(agent,
                                         bandit,
                                         logdir=logdir,
                                         log_mode=log_mode)

    def train(self, timesteps: int, **kwargs) -> None:
        """Train the agent.
        Args:
            timesteps (int, optional): Number of timesteps to train for. Defaults to 10_000.
            update_interval (int, optional): Number of timesteps between each successive
                parameter update of the agent. Defaults to 20.
            update_after (int, optional): Number of initial timesteps to start updating
                the agent's parameters after. Defaults to 500.
            batch_size (int, optional): Size of batch to update the agent with. Defaults to 64.
            train_epochs (int, optional): Number of epochs to train agent's model for in
                each update. Defaults to 20.
            log_every (int, optional): Timesteps interval for logging. Defaults to 100.
            ignore_init (int, optional): Number of initial steps to ignore for logging. Defaults to 0.
            init_train_epochs (Optional[int], optional): Initial number of epochs to train agents
                for. Defaults to None which implies `train_epochs` is to be used.
            train_epochs_decay_steps (Optional[int], optional): Steps to decay number of epochs
                to train agent for over. Defaults to None.
        Returns:
            dict: Dictionary of metrics recorded during training.
        """

        update_interval = kwargs.get("update_interval", 20)
        update_after = kwargs.get("update_after", 500)
        train_epochs = kwargs.get("train_epochs", 20)
        log_every = kwargs.get("log_every", 100)
        ignore_init = kwargs.get("ignore_init", 0)
        init_train_epochs = kwargs.get("init_train_epochs", None)
        train_epochs_decay_steps = kwargs.get("train_epochs_decay_steps", None)

        start_time = datetime.now()
        print(
            f"\nStarted at {start_time:%d-%m-%y %H:%M:%S}\n"
            f"Training {self.agent.__class__.__name__} on {self.bandit.__class__.__name__} "
            f"for {timesteps} timesteps")
        mv_len = timesteps // 20
        context = self.bandit.reset()
        regret_mv_avgs = []
        reward_mv_avgs = []

        train_epochs_schedule = None
        if init_train_epochs is not None and train_epochs_decay_steps is not None:
            train_epochs_schedule = np.linspace(init_train_epochs,
                                                train_epochs,
                                                train_epochs_decay_steps)

        for t in range(1, timesteps + 1):
            action = self.agent.select_action(context)
            new_context, reward = self.bandit.step(action)
            self.agent.update_db(context, action, reward)
            context = new_context

            if train_epochs_schedule is not None and t < train_epochs_decay_steps:
                train_epochs = int(train_epochs_schedule[t])

            if t > update_after and t % update_interval == 0:
                self.agent.update_params(action, kwargs.get("batch_size", 64),
                                         train_epochs)

            if t > ignore_init:
                regret_mv_avgs.append(
                    np.mean(self.bandit.regret_hist[-mv_len:]))
                reward_mv_avgs.append(
                    np.mean(self.bandit.reward_hist[-mv_len:]))
                if t % log_every == 0:
                    self.logger.write({
                        "timestep":
                        t,
                        "regret/regret":
                        self.bandit.regret_hist[-1],
                        "reward/reward":
                        reward,
                        "regret/cumulative_regret":
                        self.bandit.cum_regret,
                        "reward/cumulative_reward":
                        self.bandit.cum_reward,
                        "regret/regret_moving_avg":
                        regret_mv_avgs[-1],
                        "reward/reward_moving_avg":
                        reward_mv_avgs[-1],
                    })

        self.logger.close()
        print(
            f"Training completed in {(datetime.now() - start_time).seconds} seconds\n"
            f"Final Regret Moving Average: {regret_mv_avgs[-1]} | "
            f"Final Reward Moving Average: {reward_mv_avgs[-1]}")

        return {
            "regrets": self.bandit.regret_hist,
            "rewards": self.bandit.reward_hist,
            "cumulative_regrets": self.bandit.cum_regret_hist,
            "cumulative_rewards": self.bandit.cum_reward_hist,
            "regret_moving_avgs": regret_mv_avgs,
            "reward_moving_avgs": reward_mv_avgs,
        }
