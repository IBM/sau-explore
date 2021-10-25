from typing import Tuple

import numpy as np
import pandas as pd
import torch

from .bandits import DataBasedBandit
from .utils import download_data, fetch_data_without_header, get_values

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"


class MushroomDataBandit(DataBasedBandit):
    """A contextual bandit based on the `Mushroom Classification Dataset` from the UCI Machine Learning Repository.

    Source:
        https://archive.ics.uci.edu/ml/datasets/Mushroom

    Citation:
        Schlimmer. Mushroom records drawn from the audubon society field guide to North American mushrooms. GH Lincoff (Pres), New York, 1981.

    License:
        Provided by UCI Machine Learning under Open Data Commons


    Args:
        path (str, optional): Path to the data. Defaults to "./data/Mushroom/".
        download (bool, optional): Whether to download the data. Defaults to False.
        force_download (bool, optional): Whether to force download even if file exists.
            Defaults to False.
        url (Union[str, None], optional): URL to download data from. Defaults to None
            which implies use of source URL.
        r_pass (int): Reward generated for passing. Defaults to 0
        r_edible (int): Reward generated for eating an edible mushroom. Defaults to 5
        r_poisonous_lucky (int): Reward generated for eating a poisonous mushroom
            and getting lucky. Defaults to 5
        r_poisonous_unlucky (int): Reward generated for eating a poisonous mushroom
            and getting unlucky. Defaults to -35
        lucky_prob (float): Probability with which you can get lucky when eating a
            poisonous mushroom. Defaults to 0.5
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".

    Attributes:
        n_actions (int): Number of actions available.
        context_dim (int): The length of context vector.
        device (torch.device): Device to use for tensor operations.

    Raises:
        FileNotFoundError: If file is not found at specified path.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super(MushroomDataBandit, self).__init__(kwargs.get("device", "cpu"))

        path = kwargs.get("path", "./data/Mushroom/")
        download = kwargs.get("download", None)
        force_download = kwargs.get("force_download", None)
        url = kwargs.get("url", URL)

        if download:
            fpath = download_data(path, url, force_download)
            self.df = pd.read_csv(fpath, header=None)
        else:
            self.df = fetch_data_without_header(path, "agaricus-lepiota.data")

        # What type of task/reward? bandit or classification?
        #  self.task = kwargs.get("task", "bandit")
        self.task = kwargs.get("task", "classification")

        # One hot encoding of categorical features
        self.df = pd.get_dummies(self.df, columns=self.df.columns)

        if self.task == "bandit":
            self.r_pass = kwargs.get("r_pass", 0)
            self.r_edible = kwargs.get("r_edible", 5)
            self.r_poisonous_lucky = kwargs.get("r_poisonous_lucky", 5)
            self.r_poisonous_unlucky = kwargs.get("r_poisonous_unlucky", -35)
            self.lucky_prob = kwargs.get("lucky_prob", 0.5)
            self.r_poisonous_exp = (
                self.r_poisonous_lucky * self.lucky_prob +
                self.r_poisonous_unlucky * (1 - self.lucky_prob)
            )
            self.n_actions = 2

        elif self.task == "classification":
            """Just learn to predict whether the mushroom is edible or poisonous
            """
            self.labels, self.n_actions = get_values(self.df.iloc[:, 0])
        else:
            raise ValueError('Task {} is unkown'.format(self.task))

        self.context_dim = self.df.shape[1] - 2
        self.len = len(self.df)

    def reset(self) -> torch.Tensor:
        """Reset bandit by shuffling indices and get new context.

        Returns:
            torch.Tensor: Current context selected by bandit.
        """
        self._reset()
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        if self.task == "bandit":
            poison_rewards = np.random.choice(
                [self.r_poisonous_lucky, self.r_poisonous_unlucky],
                p=[self.lucky_prob, 1 - self.lucky_prob],
                size=len(self.df),
            )
            self.eat_rewards = self.r_edible * self.df.iloc[:, 0] + np.multiply(
                poison_rewards, self.df.iloc[:, 1]
            )
            self.optimal_exp_rewards = (
                self.r_edible * self.df.iloc[:, 0] +
                max(self.r_pass, self.r_poisonous_exp) * self.df.iloc[:, 1]
            )
            if self.r_pass > self.r_poisonous_exp:
                self.optimal_actions = self.df.iloc[:, 0].to_numpy()
            else:
                self.optimal_actions = np.ones(len(self.df))

        else:
            self.labels, _ = get_values(self.df.iloc[:, 0])

        return self._get_context()

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        """Compute the reward for a given action.

        Args:
            action (int): The action to compute reward for.

        Returns:
            Tuple[int, int]: reward, max_reward
        """
        if self.task == "bandit":
            if action == 0:
                r = self.r_pass
            elif action == 1:
                r = self.eat_rewards[self.idx]
            else:
                raise ValueError(f"Action {action} undefined for mushroom data bandit")
            return r, self.optimal_exp_rewards[self.idx]

        else:
            label = self.labels[self.idx]
            r = int(label == action)
            return r, 1

    def _get_context(self) -> torch.Tensor:
        """Get the vector for current selected context.

        Returns:
            torch.Tensor: Current context vector.
        """
        return torch.tensor(
            self.df.iloc[self.idx, self.n_actions:],
            device=self.device,
            dtype=torch.float,
        )
