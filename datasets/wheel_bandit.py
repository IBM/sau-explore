from typing import Tuple

import numpy as np
import torch

from .bandits import DataBasedBandit


class WheelBandit(DataBasedBandit):
    """The wheel contextual bandit from the Riquelme et al 2018 paper.

    Source:
        https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits

    Citation:
        Riquelme, Tucker, Snoek. Deep Bayesian bandits showdown: An empirical comparison of Bayesian deep networks for Thompson sampling. InProceedings ofthe 6th International Conference on Learning Representations, 2018.

    Args:
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".

    Attributes:
        n_actions (int): Number of actions available.
        context_dim (int): The length of context vector.
        len (int): The number of examples (context, reward pairs) in the dataset.
        device (torch.device): Device to use for tensor operations.
    """

    def __init__(self, delta=0.5, n_samples=2000, **kwargs):
        super(WheelBandit, self).__init__(kwargs.get("device", "cpu"))

        self.delta = delta

        self.n_actions = 5
        self.context_dim = 2
        self.len = n_samples

        self.mu = [1.2, 1.0, 50.0]
        self.sigma = 0.01

        self._sign_opt_action = {
            (1.0, 1.0): 1,
            (1.0, -1.0): 2,
            (-1.0, 1.0): 3,
            (-1.0, -1.0): 4,
        }

        self._generate_contexts()
        self._generate_rewards()

    def _generate_rewards(self):
        r_all = np.random.normal(self.mu[1], self.sigma, size=(self.len, self.n_actions))
        r_all[:,0] += self.mu[0] - self.mu[1]
        for t in range(self.len):
            if np.linalg.norm(self._context[t]) > self.delta:
                signs = np.sign(self._context[t])
                opt_act = self._sign_opt_action[(signs[0], signs[1])]
                r_all[t, opt_act] += self.mu[2] - self.mu[1]
        self.rewards = r_all
        self.max_rewards = np.max(self.rewards, axis=1)

    def reset(self) -> torch.Tensor:
        """Reset bandit by shuffling indices and get new context.

        Returns:
            torch.Tensor: Current context selected by bandit.
        """
        self._reset()
        self._generate_contexts()
        self._generate_rewards()
        return self._get_context()

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        """Compute the reward for a given action.

        Args:
            action (int): The action to compute reward for.

        Returns:
            Tuple[int, int]: Computed reward.
        """
        r = self.rewards[self.idx, action]
        max_r = self.max_rewards[self.idx]
        return r, max_r

    def _generate_contexts(self) -> None:
        """Returns 2-dim samples falling in the unit circle.
        """
        theta = np.random.uniform(0.0, 2.0 * np.pi, (self.len))
        r = np.sqrt(np.random.uniform(size=self.len))  # sqrt is in the original code of Riquelme et al
        self._context = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    def _get_context(self) -> torch.Tensor:
        """Get the vector for current selected context.

        Returns:
            torch.Tensor: Current context vector.
        """
        return torch.tensor(
            self._context[self.idx],
            device=self.device,
            dtype=torch.float,
        )
