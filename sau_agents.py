from typing import Optional

import torch

from datasets.bandits import DataBasedBandit, TransitionDB
from lgenrl import DCBAgent, RMSNeuralModel


class SAULinearAgent(DCBAgent):
    """Deep contextual bandit agent using a linear model with SAU.

    Args:
        bandit (DataBasedBandit): The bandit to solve
        strategy (str, optional): Which SAU strategy to use ('sampling'|'ucb')
            Default: 'sampling'
        init_pulls (int, optional): Number of times to select each action initially.
            Defaults to 3.
        lambda_prior (float, optional): Guassian prior for linear model. Defaults to 0.25.
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".
    """

    def __init__(self, bandit: DataBasedBandit, **kwargs):
        super(SAULinearAgent, self).__init__(bandit, kwargs.get("device", "cpu"))

        self.strategy = kwargs.get("strategy", "sampling")
        self.init_pulls = kwargs.get("init_pulls", 2)
        self.lambda_prior = kwargs.get("lambda_prior", 0.25)

        self.beta = torch.zeros(
            size=(self.n_actions, self.context_dim + 1),
            device=self.device,
            dtype=torch.float,
        )
        self.xcov = torch.zeros(
            size=(self.n_actions, self.context_dim + 1),
            device=self.device,
            dtype=torch.float,
        )
        self.inv_cov = torch.stack(
            [
                self.lambda_prior
                * torch.eye(self.context_dim + 1, device=self.device, dtype=torch.float)
                for _ in range(self.n_actions)
            ]
        )

        self.db = TransitionDB(self.device)
        self.t = 0
        self.update_count = 0

        self.n_a = torch.zeros(self.n_actions)  # Actions count
        self.n_a_last_udpate = self.n_actions * [0]
        self.sau2 = torch.ones(
            self.n_actions
        )  # Sample Average Uncertainty, initialized at 1 (sau2_a = n_a * tau2_a)

    def pred_rewards(self, context: torch.Tensor) -> torch.Tensor:
        """Predict rewards with exploration bonus using SAU-UCB"""
        values = torch.mv(self.beta, torch.cat([context.view(-1), torch.ones(1)]))

        if self.strategy == "ucb":
            rewards_sigma = (self.sau2 * self.n_a.sum().log()).sqrt() / self.n_a
            pred_rewards = values + rewards_sigma
        elif self.strategy == "sampling":
            rewards_sigma = self.sau2.sqrt() / self.n_a
            pred_rewards = values + rewards_sigma * torch.randn_like(values)
        else:
            raise ValueError(f"Stratey {self.strategy} undefined for SAULinearAgent")
        return pred_rewards

    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on given context.

        Selecting action with highest predicted reward computed through
        betas sampled from posterior.

        Args:
            context (torch.Tensor): The context vector to select action for.

        Returns:
            int: The action to take.
        """
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(self.t % self.n_actions, device=self.device, dtype=torch.int)

        pred_rewards = self.pred_rewards(context)
        action = torch.argmax(pred_rewards).to(torch.int)
        return action

    def update_db(self, context: torch.Tensor, action: int, reward: int):
        """Updates transition database with given transition

        Args:
            context (torch.Tensor): Context received
            action (int): Action taken
            reward (int): Reward received
        """
        self.db.add(context, action, reward)
        self.n_a[action] += 1
        pred_rewards = self.pred_rewards(context)
        delta = reward - pred_rewards[action]
        self.sau2[action] += delta * delta

    def update_params(
        self, action: int, batch_size: int = 512, train_epochs: Optional[int] = None
    ):
        """Update parameters of the agent.

        Updated the posterior over beta though bayesian regression.

        Args:
            action (int): Action to update the parameters for.
            batch_size (int, optional): Size of batch to update parameters with.
                Defaults to 512
            train_epochs (Optional[int], optional): Epochs to train neural network for.
                Not applicable in this agent. Defaults to None
        """
        self.update_count += 1

        n_a0 = self.n_a_last_udpate[action]
        n_d = int(self.n_a[action].item()) - n_a0
        self.n_a_last_udpate[action] = int(self.n_a[action].item())

        x, y = self.db.get_data_for_action(action)
        x, y = x[-n_d:], y[-n_d:]
        x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)

        self.inv_cov[action] = torch.mm(x.T, x) + self.inv_cov[action]
        cov = torch.inverse(self.inv_cov[action] / (n_a0 + 1)) / (n_a0 + 1)
        self.xcov[action] = torch.mm(x.T, y).squeeze() + self.xcov[action]

        beta = torch.mm(cov, self.xcov[action].unsqueeze(1))
        self.beta[action] = beta.squeeze(1)


class SAUNeuralAgent(DCBAgent):
    """Deep contextual bandit agent based on a neural network using SAU to estimate uncertainty.

    Args:
        bandit (DataBasedBandit): The bandit to solve
        strategy (str, optional): Which SAU strategy to use ('sampling'|'ucb')
            Default: 'sampling'
        init_pulls (int, optional): Number of times to select each action initially.
            Default: 1
        hidden_dims (List[int], optional): Dimensions of hidden layers of network.
            Default: [50, 50]
        init_lr (float, optional): Initial learning rate.
            Default: 0.1
        lr_decay (float, optional): Decay rate for learning rate.
            Default: 0.0
        lr_reset (bool, optional): Whether to reset learning rate ever train interval.
            Default: True
        dropout_p (Optional[float], optional): Probability for dropout.
            Defaults to None (dropout is not to be used)
        eval_with_dropout (bool, optional): Whether or not to use dropout at inference.
            Default: False
    """

    def __init__(self, bandit: DataBasedBandit, **kwargs):
        super(SAUNeuralAgent, self).__init__(bandit, kwargs.get("device", "cpu"))

        self.strategy = kwargs.get("strategy", "sampling")
        self.init_pulls = kwargs.get("init_pulls", 2)
        self.eval_with_dropout = kwargs.get("eval_with_dropout", False)
        hidden_dims = kwargs.get("hidden_dims", [100, 100])
        init_lr = kwargs.get("init_lr", 0.005)
        lr_decay = kwargs.get("lr_decay", 0.5)
        lr_reset = kwargs.get("lr_reset", False)
        dropout_p = kwargs.get("dropout_p", None)

        self.model = (
            RMSNeuralModel(
                context_dim=self.context_dim,
                hidden_dims=hidden_dims,
                n_actions=self.n_actions,
                init_lr=init_lr,
                lr_decay=lr_decay,
                lr_reset=lr_reset,
                dropout_p=dropout_p,
            )
            .to(torch.float)
            .to(self.device)
        )
        self.db = TransitionDB(self.device)

        self.t = 0
        self.update_count = 0

        self.n_a = torch.zeros(self.n_actions)  # Actions count
        self.sau2 = torch.ones(
            self.n_actions
        )  # Sample Average Uncertainty (sau2_a = n_a * tau2_a)

    def select_action(self, context: torch.Tensor) -> int:
        """Selects action for a given context"""
        self.model.use_dropout = self.eval_with_dropout
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            action = self.t % self.n_actions
            return torch.tensor(action, device=self.device, dtype=torch.int)

        results = self.pred_rewards(context)
        action = torch.argmax(results["pred_rewards"]).to(torch.int)
        return action

    def pred_rewards(self, context: torch.Tensor) -> torch.Tensor:
        """Predict rewards with exploration bonus using SAU-UCB"""
        re = self.model(context)
        re["rewards_mu"] = re["pred_rewards"]

        if self.strategy == "ucb":
            re["rewards_sigma"] = (self.sau2 * self.n_a.sum().log()).sqrt() / self.n_a
            re["pred_rewards"] = re["rewards_mu"] + re["rewards_sigma"]
        elif self.strategy == "sampling":
            re["rewards_sigma"] = self.sau2.sqrt() / self.n_a
            re["pred_rewards"] = re["rewards_mu"] + re["rewards_sigma"] * torch.randn_like(
                re["rewards_mu"]
            )
        else:
            raise ValueError(f"Stratey {self.strategy} undefined for SAUNeuralAgent")
        return re

    def update_db(self, context: torch.Tensor, action: int, reward: int):
        """Updates transition database, and SAU tau"""
        self.db.add(context, action, reward)
        self.n_a[action] += 1
        results = self.pred_rewards(context)
        delta = reward - results["rewards_mu"][action]
        self.sau2[action] += delta * delta

    def update_params(
        self, action: Optional[int] = None, batch_size: int = 512, train_epochs: int = 20
    ):
        """Update parameters of the agent."""
        self.update_count += 1
        self.model.train_model(self.db, train_epochs, batch_size)
