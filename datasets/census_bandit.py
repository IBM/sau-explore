from typing import Tuple

import numpy as np
import pandas as pd
import torch

from .bandits import DataBasedBandit
from .utils import download_data, fetch_data_with_header
from .utils import get_values

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"


class CensusDataBandit(DataBasedBandit):
    """A contextual bandit based on the `US Census Data (1990)` dataset from the UCI Machine Learning Repository.

    Source:
        https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29

    Citation:
        Meek, Thiesson, and  Heckerman. "The learning-curve sampling method applied to model-based clustering." Journal of Machine Learning Research, 2002

    License:
        Provided by UCI Machine Learning under Open Data Commons


    Args:
        path (str, optional): Path to the data. Defaults to "./data/Census/".
        download (bool, optional): Whether to download the data. Defaults to False.
        force_download (bool, optional): Whether to force download even if file exists.
            Defaults to False.
        url (Union[str, None], optional): URL to download data from. Defaults to None
            which implies use of source URL.
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".


    Attributes:
        n_actions (int): Number of actions available.
        context_dim (int): The length of context vector.
        len (int): The number of examples (context, reward pairs) in the dataset.
        device (torch.device): Device to use for tensor operations.

    Raises:
        FileNotFoundError: If file is not found at specified path.
    """

    def __init__(self, **kwargs):
        super(CensusDataBandit, self).__init__(kwargs.get("device", "cpu"))

        path = kwargs.get("path", "./data/Census/")
        download = kwargs.get("download", None)
        force_download = kwargs.get("force_download", None)
        url = kwargs.get("url", URL)

        if download:
            fpath = download_data(path, url, force_download)
            df = pd.read_csv(fpath)
        else:
            df = fetch_data_with_header(path, "USCensus1990.data.txt")

        self.labels, self.n_actions = get_values(df["dOccup"])
        df = df.drop(["dOccup", "caseid"], axis=1)

        self.X = pd.get_dummies(df, columns=df.columns).to_numpy()
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
