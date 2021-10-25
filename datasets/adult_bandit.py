from typing import Tuple

import numpy as np
import pandas as pd
import torch

from .bandits import DataBasedBandit
from .utils import download_data, fetch_data_without_header
from .utils import normalize_cat_num, get_values

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"


class AdultDataBandit(DataBasedBandit):
    """A contextual bandit based on the `Adult dataset` from the UCI Machine Learning Repository.

    Source:
        http://archive.ics.uci.edu/ml/datasets/Adult

    Citation:
        Kohavi, “Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid”, Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996

    License:
        Provided by UCI Machine Learning under Open Data Commons

    Args:
        path (str, optional): Path to the data. Defaults to "./data/Adult/".
        download (bool, optional): Whether to download the data. Defaults to False.
        force_download (bool, optional): Whether to force download even if file exists.
            Defaults to False.
        url (Union[str, None], optional): URL to download data from. Defaults to None
            which implies use of source URL.
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".

    Attributes:
        n_actions (int): Number of actions available, 14
        context_dim (int): The length of context vector.
        len (int): The number of examples (context, reward pairs) in the dataset.
        device (torch.device): Device to use for tensor operations.

    Raises:
        FileNotFoundError: If file is not found at specified path.
    """

    def __init__(self, **kwargs):
        super(AdultDataBandit, self).__init__(kwargs.get("device", "cpu"))

        path = kwargs.get("path", "./data/Adult/")
        download = kwargs.get("download", None)
        force_download = kwargs.get("force_download", None)
        url = kwargs.get("url", URL)

        if download:
            fpath = download_data(path, url, force_download)
            df = pd.read_csv(fpath, header=None, na_values=["?", " ?"]).dropna()
        else:
            df = fetch_data_without_header(
                path, "adult.data", na_values=["?", " ?"]
            )

        self.labels, self.n_actions = get_values(df[6])
        df = df.drop([6], axis=1)

        cols_cat = [1, 3, 5, 7, 8, 9, 13, 14]
        self.X = normalize_cat_num(df, cols_cat)

        self.len, self.context_dim = self.X.shape

    def reset(self) -> torch.Tensor:
        """Reset bandit by shuffling indices and get new context.

        Returns:
            torch.Tensor: Current context selected by bandit.
        """
        self._reset()
        perm = np.random.permutation(self.len)
        self.X = self.X[perm]
        self.labels = self.labels[perm]
        return self._get_context()

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        """Compute the reward for a given action.

        Args:
            action (int): The action to compute reward for.

        Returns:
            Tuple[int, int]: Computed reward.
        """
        label = self.labels[self.idx]
        r = int(label == action)
        return r, 1

    def _get_context(self) -> torch.Tensor:
        """Get the vector for current selected context.

        Returns:
            torch.Tensor: Current context vector.
        """
        return torch.tensor(
            self.X[self.idx],
            device=self.device,
            dtype=torch.float,
        )
