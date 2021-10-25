from typing import List, Optional

import numpy as np
import torch
from scipy.stats import invgamma

from datasets.bandits import DataBasedBandit, TransitionDB
from lgenrl import DCBAgent, RMSNeuralModel


class LinearUCBAgent(DCBAgent):
    """Deep contextual bandit agent using bayesian regression for posterior inference.

    Args:
        bandit (DataBasedBandit): The bandit to solve
        init_pulls (int, optional): Number of times to select each action initially.
            Defaults to 3.
        lambda_prior (float, optional): Guassian prior for linear model. Defaults to 0.25.
        a0 (float, optional): Inverse gamma prior for noise. Defaults to 6.0.
        b0 (float, optional): Inverse gamma prior for noise. Defaults to 6.0.
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".
    """

    def __init__(self, bandit: DataBasedBandit, **kwargs):
        super(LinearUCBAgent, self).__init__(bandit, kwargs.get("device", "cpu"))

        self.init_pulls = kwargs.get("init_pulls", 2)
        self.lambda_prior = kwargs.get("lambda_prior", 0.25)
        self.a0 = kwargs.get("a0", 6.0)
        self.b0 = kwargs.get("b0", 6.0)
        self.mu = torch.zeros(
            size=(self.n_actions, self.context_dim + 1),
            device=self.device,
            dtype=torch.float,
        )
        self.xcov = torch.zeros(
            size=(self.n_actions, self.context_dim + 1),
            device=self.device,
            dtype=torch.float,
        )
        self.ycov = torch.zeros(
            self.n_actions,
            device=self.device,
            dtype=torch.float,
        )
        self.cov = torch.stack(
            [
                (1.0 / self.lambda_prior)
                * torch.eye(self.context_dim + 1, device=self.device, dtype=torch.float)
                for _ in range(self.n_actions)
            ]
        )
        self.inv_cov = torch.stack(
            [
                self.lambda_prior
                * torch.eye(self.context_dim + 1, device=self.device, dtype=torch.float)
                for _ in range(self.n_actions)
            ]
        )
        self.a = self.a0 * torch.ones(self.n_actions, device=self.device, dtype=torch.float)
        self.b = self.b0 * torch.ones(self.n_actions, device=self.device, dtype=torch.float)
        self.db = TransitionDB(self.device)
        self.t = 0
        self.update_count = 0

        self.n_a = torch.zeros(self.n_actions)  # Actions count
        self.n_a_last_udpate = self.n_actions * [0]

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
            action = self.t % self.n_actions
            return torch.tensor(action, device=self.device, dtype=torch.int)
        var = torch.tensor(
            [self.b[i] * invgamma.rvs(self.a[i]) for i in range(self.n_actions)],
            device=self.device,
            dtype=torch.float,
        )
        try:
            beta = (
                torch.tensor(
                    np.stack(
                        [
                            np.random.multivariate_normal(self.mu[i], var[i] * self.cov[i])
                            for i in range(self.n_actions)
                        ]
                    )
                )
                .to(self.device)
                .to(torch.float)
            )
        except np.linalg.LinAlgError as e:  # noqa F841
            beta = (
                (
                    torch.stack(
                        [
                            torch.distributions.MultivariateNormal(
                                torch.zeros(self.context_dim + 1),
                                torch.eye(self.context_dim + 1),
                            ).sample()
                            for i in range(self.n_actions)
                        ]
                    )
                )
                .to(self.device)
                .to(torch.float)
            )
        values = torch.mv(beta, torch.cat([context.view(-1), torch.ones(1)]))
        action = torch.argmax(values).to(torch.int)
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

        inv_cov = torch.mm(x.T, x) + self.inv_cov[action]
        self.cov[action] = torch.inverse(inv_cov / (n_a0 + 1)) / (n_a0 + 1)
        self.xcov[action] = torch.mm(x.T, y).squeeze() + self.xcov[action]
        self.ycov[action] = torch.mm(y.T, y) + self.ycov[action]

        mu = torch.mm(self.cov[action], self.xcov[action].unsqueeze(1))
        self.a[action] = self.a0 + self.n_a[action] / 2
        self.b[action] = self.b0 + (self.ycov[action] - torch.mm(mu.T, torch.mm(inv_cov, mu))) / 2
        self.mu[action] = mu.squeeze(1)
        self.inv_cov[action] = inv_cov


class NeuralLinearUCBAgent(DCBAgent):
    """Deep contextual bandit agent using bayesian regression on for posterior inference

    A neural network is used to transform context vector to a latent representation on
    which bayesian regression is performed.
    Contrary to the original genrl implementation, there is not latent_db DB, and the
    latent representations are recomputed on the fly on the current NN weights

    Args:
        bandit (DataBasedBandit): The bandit to solve
        init_pulls (int, optional): Number of times to select each action initially.
            Defaults to 3.
        hidden_dims (List[int], optional): Dimensions of hidden layers of network.
            Defaults to [50, 50].
        init_lr (float, optional): Initial learning rate. Defaults to 0.1.
        lr_decay (float, optional): Decay rate for learning rate. Defaults to 0.5.
        lr_reset (bool, optional): Whether to reset learning rate ever train interval.
            Defaults to True.
        max_grad_norm (float, optional): Maximum norm of gradients for gradient clipping.
            Defaults to 0.5.
        dropout_p (Optional[float], optional): Probability for dropout. Defaults to None
            which implies dropout is not to be used.
        eval_with_dropout (bool, optional): Whether or not to use dropout at inference.
            Defaults to False.
        nn_update_ratio (int, optional): . Defaults to 2.
        lambda_prior (float, optional): Guassian prior for linear model. Defaults to 0.25.
        a0 (float, optional): Inverse gamma prior for noise. Defaults to 3.0.
        b0 (float, optional): Inverse gamma prior for noise. Defaults to 3.0.
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".
    """

    def __init__(self, bandit: DataBasedBandit, **kwargs):
        super(NeuralLinearUCBAgent, self).__init__(bandit, kwargs.get("device", "cpu"))
        self.init_pulls = kwargs.get("init_pulls", 2)
        self.eval_with_dropout = kwargs.get("eval_with_dropout", False)
        self.lambda_prior = kwargs.get("lambda_prior", 0.25)
        self.a0 = kwargs.get("a0", 6.0)
        self.b0 = kwargs.get("b0", 6.0)
        hidden_dims = kwargs.get("hidden_dims", [100, 100])
        init_lr = kwargs.get("init_lr", 0.005)
        lr_decay = kwargs.get("lr_decay", 0.5)
        lr_reset = kwargs.get("lr_reset", False)
        dropout_p = kwargs.get("dropout_p", None)
        self.latent_dim = hidden_dims[-1]
        self.nn_update_ratio = kwargs.get("nn_update_ratio", 2)

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
        self.mu = torch.zeros(
            size=(self.n_actions, self.latent_dim + 1),
            device=self.device,
            dtype=torch.float,
        )
        self.xcov = torch.zeros(
            size=(self.n_actions, self.latent_dim + 1),
            device=self.device,
            dtype=torch.float,
        )
        self.ycov = torch.zeros(
            self.n_actions,
            device=self.device,
            dtype=torch.float,
        )
        self.cov = torch.stack(
            [
                (1.0 / self.lambda_prior)
                * torch.eye(self.latent_dim + 1, device=self.device, dtype=torch.float)
                for _ in range(self.n_actions)
            ]
        )
        self.inv_cov = torch.stack(
            [
                self.lambda_prior
                * torch.eye(self.latent_dim + 1, device=self.device, dtype=torch.float)
                for _ in range(self.n_actions)
            ]
        )
        self.a = self.a0 * torch.ones(self.n_actions, device=self.device, dtype=torch.float)
        self.b = self.b0 * torch.ones(self.n_actions, device=self.device, dtype=torch.float)
        self.db = TransitionDB(self.device)
        self.t = 0
        self.n_a = torch.zeros(self.n_actions)  # Actions count
        self.n_a_last_udpate = self.n_actions * [0]
        self.update_count = 0

    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on given context.

        Selects an action by computing a forward pass through network to output
        a representation of the context on which bayesian linear regression is
        performed to select an action.

        Args:
            context (torch.Tensor): The context vector to select action for.

        Returns:
            int: The action to take.
        """
        self.model.use_dropout = self.eval_with_dropout
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(self.t % self.n_actions, device=self.device, dtype=torch.int)
        var = torch.tensor(
            [self.b[i] * invgamma.rvs(self.a[i]) for i in range(self.n_actions)],
            device=self.device,
            dtype=torch.float,
        )
        try:
            beta = (
                torch.tensor(
                    np.stack(
                        [
                            np.random.multivariate_normal(self.mu[i], var[i] * self.cov[i])
                            for i in range(self.n_actions)
                        ]
                    )
                )
                .to(self.device)
                .to(torch.float)
            )
        except np.linalg.LinAlgError as e:  # noqa F841
            beta = (
                (
                    torch.stack(
                        [
                            torch.distributions.MultivariateNormal(
                                torch.zeros(self.latent_dim + 1),
                                torch.eye(self.latent_dim + 1),
                            ).sample()
                            for i in range(self.n_actions)
                        ]
                    )
                )
                .to(self.device)
                .to(torch.float)
            )
        results = self.model(context)
        latent_context = results["x"]
        values = torch.mv(beta, torch.cat([latent_context.squeeze(0), torch.ones(1)]))
        action = torch.argmax(values).to(torch.int)
        return action

    def update_db(self, context: torch.Tensor, action: int, reward: int):
        """Updates transition database with given transition

        Updates latent context and predicted rewards seperately.

        Args:
            context (torch.Tensor): Context received
            action (int): Action taken
            reward (int): Reward received
        """
        self.db.add(context, action, reward)
        self.n_a[action] += 1

    def update_params(self, action: int, batch_size: int = 512, train_epochs: int = 20):
        """Update parameters of the agent.

        Trains neural network and updates bayesian regression parameters.

        Args:
            action (int): Action to update the parameters for.
            batch_size (int, optional): Size of batch to update parameters with.
                Defaults to 512
            train_epochs (int, optional): Epochs to train neural network for.
                Defaults to 20
        """
        self.update_count += 1

        if self.update_count % self.nn_update_ratio == 0:
            self.model.train_model(self.db, train_epochs, batch_size)

        n_a0 = self.n_a_last_udpate[action]
        n_d = int(self.n_a[action].item()) - n_a0
        self.n_a_last_udpate[action] = int(self.n_a[action].item())

        x, y = self.db.get_data_for_action(action)
        x, y = x[-n_d:], y[-n_d:]

        results = self.model(x)
        z = results["x"].detach()
        z = torch.cat([z, torch.ones(z.shape[0], 1)], dim=1)

        inv_cov = torch.mm(z.T, z) + self.inv_cov[action]
        self.cov[action] = torch.inverse(inv_cov / (n_a0 + 1)) / (n_a0 + 1)
        self.xcov[action] = torch.mm(z.T, y).squeeze() + self.xcov[action]
        self.ycov[action] = torch.mm(y.T, y) + self.ycov[action]

        mu = torch.mm(self.cov[action], self.xcov[action].unsqueeze(1))
        self.a[action] = self.a0 + self.n_a[action] / 2
        self.b[action] = self.b0 + (self.ycov[action] - torch.mm(mu.T, torch.mm(inv_cov, mu))) / 2
        self.mu[action] = mu.squeeze(1)
        self.inv_cov[action] = inv_cov


class LinearGreedyAgent(DCBAgent):
    """Contextual bandit agent using a greedy linear model.

    Args:
        bandit (DataBasedBandit): The bandit to solve
        init_pulls (int, optional): Number of times to select each action initially.
            Defaults to 3.
        lambda_prior (float, optional): Guassian prior for linear model. Defaults to 0.25.
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".
    """

    def __init__(self, bandit: DataBasedBandit, **kwargs):
        super(LinearGreedyAgent, self).__init__(bandit, kwargs.get("device", "cpu"))
        self.epsilon = kwargs.get("epsilon", 0.046)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.93)

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
        if torch.rand(1) < self.epsilon:
            action = torch.randint(self.n_actions, size=(1,)).to(torch.int).squeeze()
        else:
            pred_rewards = torch.mv(self.beta, torch.cat([context.view(-1), torch.ones(1)]))
            action = torch.argmax(pred_rewards).to(torch.int).squeeze()
        return action

    def update_db(self, context: torch.Tensor, action: int, reward: int):
        """Updates transition database with given transition

        Args:
            context (torch.Tensor): Context received
            action (int): Action taken
            reward (int): Reward received
        """
        self.db.add(context, action, reward)
        self.epsilon = self.epsilon * self.epsilon_decay
        self.n_a[action] += 1

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


class NeuralGreedyAgent(DCBAgent):
    """Deep contextual bandit agent using epsilon greedy with a neural network.

    Args:
        bandit (DataBasedBandit): The bandit to solve
        init_pulls (int, optional): Number of times to select each action initially.
            Defaults to 3.
        hidden_dims (List[int], optional): Dimensions of hidden layers of network.
            Defaults to [50, 50].
        init_lr (float, optional): Initial learning rate. Defaults to 0.1.
        lr_decay (float, optional): Decay rate for learning rate. Defaults to 0.5.
        lr_reset (bool, optional): Whether to reset learning rate ever train interval.
            Defaults to True.
        max_grad_norm (float, optional): Maximum norm of gradients for gradient clipping.
            Defaults to 0.5.
        dropout_p (Optional[float], optional): Probability for dropout. Defaults to None
            which implies dropout is not to be used.
        eval_with_dropout (bool, optional): Whether or not to use dropout at inference.
            Defaults to False.
        epsilon (float, optional): Probability of selecting a random action. Defaults to 0.0.
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".
    """

    def __init__(self, bandit: DataBasedBandit, **kwargs):
        super(NeuralGreedyAgent, self).__init__(bandit, kwargs.get("device", "cpu"))
        self.epsilon = kwargs.get("epsilon", 0.046)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.93)

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

    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on given context.

        Selects an action by computing a forward pass through network
        with an epsillon probability of selecting a random action.

        Args:
            context (torch.Tensor): The context vector to select action for.

        Returns:
            int: The action to take.
        """
        self.model.use_dropout = self.eval_with_dropout
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(self.t % self.n_actions, device=self.device, dtype=torch.int).view(
                1
            )
        if torch.rand(1) < self.epsilon:
            action = torch.randint(self.n_actions, size=(1,)).to(torch.int)
        else:
            results = self.model(context)
            action = torch.argmax(results["pred_rewards"]).to(torch.int).view(1)
        return action

    def update_db(self, context: torch.Tensor, action: int, reward: int):
        """Updates transition database with given transition

        Args:
            context (torch.Tensor): Context received
            action (int): Action taken
            reward (int): Reward received
        """
        self.db.add(context, action, reward)
        self.epsilon = self.epsilon * self.epsilon_decay

    def update_params(
        self,
        action: Optional[int] = None,
        batch_size: int = 512,
        train_epochs: int = 20,
    ):
        """Update parameters of the agent.

        Trains neural network.

        Args:
            action (Optional[int], optional): Action to update the parameters for.
                Not applicable in this agent. Defaults to None.
            batch_size (int, optional): Size of batch to update parameters with.
                Defaults tp 512
            train_epochs (int, optional): Epochs to train neural network for.
                Defaults to 20
        """
        self.update_count += 1
        self.model.train_model(self.db, train_epochs, batch_size)


class UniformAgent(DCBAgent):
    def __init__(self, bandit: DataBasedBandit, p: List[float] = None, device: str = "cpu"):
        """A fixed policy agent for deep contextual bandits.

        Args:
            bandit (DataBasedBandit): Bandit to solve.
            p (List[float], optional): List of probabilities for each action.
                Defaults to None which implies action is sampled uniformly.
            device (str): Device to use for tensor operations.
                "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".

        Raises:
            ValueError: Raised if length of given probabilities is not
                equal to the number of actions available in given bandit.
        """
        super(UniformAgent, self).__init__(bandit, device)
        if p is None:
            p = [1 / self.n_actions for _ in range(self.n_actions)]
        elif len(p) != self.n_actions:
            raise ValueError(f"p should be of length {self.n_actions}")
        self.p = p
        self.t = 0

    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on fixed probabilities.

        Args:
            context (torch.Tensor): The context vector to select action for.
                In this agent, context vector is not considered.

        Returns:
            int: The action to take.
        """
        self.t += 1
        return np.random.choice(range(self.n_actions), p=self.p)

    def update_db(self, *args, **kwargs):
        pass

    def update_params(self, *args, **kwargs):
        pass
