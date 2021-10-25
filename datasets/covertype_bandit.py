from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from .bandits import DataBasedBandit
from .utils import download_data, fetch_data_without_header
from .utils import safe_zscore, get_values

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
)


class CovertypeDataBandit(DataBasedBandit):
    """A contextual bandit based on the `Forest Cover Type Dataset` from the UCI Machine Learning Repository.

    Source:
        https://archive.ics.uci.edu/ml/datasets/covertype

    Citation:
        Blackard, Jock A. 1998. "Comparison of Neural Networks and Discriminant Analysis in Predicting Forest Cover Types." Ph.D. dissertation. Department of Forest Sciences. Colorado State University. Fort Collins, Colorado.

    License:
        Provided by UCI Machine Learning under Open Data Commons


    Args:
        path (str, optional): Path to the data. Defaults to "./data/Covertype/".
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
        super(CovertypeDataBandit, self).__init__(kwargs.get("device", "cpu"))

        path = Path(kwargs.get("path", "./data/Covertype/"))
        download = kwargs.get("download", None)
        force_download = kwargs.get("force_download", None)
        url = kwargs.get("url", URL)

        if download:
            fpath = download_data(path, url, force_download)
            df = pd.read_csv(fpath, header=None, na_values=["?"]).dropna()
        else:
            df = fetch_data_without_header(path, "covtype.data.gz")

        self.labels, self.n_actions = get_values(df.iloc[:, -1])

        self.X = safe_zscore(df.iloc[:, :-1].to_numpy())
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
