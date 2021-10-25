import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from .bandits import DataBasedBandit
from .utils import download_data, fetch_data_without_header
from .utils import safe_zscore, get_values

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z"


class StatlogDataBandit(DataBasedBandit):
    """A contextual bandit based on the `Statlog (Shuttle) dataset` from the UCI Machine Learning Repository.

    Source:
        https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)

    Citation:
        Asuncion and Newman. UCI machine learning repository, 2007.

    License:
        Provided by UCI Machine Learning under Open Data Commons


    Desciption:
        The dataset gives the recorded value of 9 different sensors during a space shuttle flight
        as well which state (out of 7 possible) the radiator was at each timestep.

        At each timestep the agent will get a 9-dimensional real valued context vector
        and must select one of 7 actions. The agent will get a reward of 1 only if it selects
        the true state of the radiator at that timestep as given in the dataset.

        Context dimension: 9
        Number of actions: 7


    Args:
        path (str, optional): Path to the data. Defaults to "./data/Statlog/".
        download (bool, optional): Whether to download the data. Defaults to False.
        force_download (bool, optional): Whether to force download even if file exists.
            Defaults to False.
        url (Union[str, None], optional): URL to download data from. Defaults to None
            which implies use of source URL.w
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".

    Attributes:
        n_actions (int): Number of actions available.
        context_dim (int): The length of context vector.
        device (torch.device): Device to use for tensor operations.

    Raises:
        FileNotFoundError: If file is not found at specified path.
    """

    def __init__(self, **kwargs):
        super(StatlogDataBandit, self).__init__(kwargs.get("device", "cpu"))

        path = kwargs.get("path", "./data/Statlog/")
        download = kwargs.get("download", None)
        force_download = kwargs.get("force_download", None)
        url = kwargs.get("url", URL)

        file_exists = Path(path).joinpath("shuttle.trn").is_file()
        if not download or (file_exists and not force_download):
            df = fetch_data_without_header(path, "shuttle.trn", delimiter=" ")
        else:
            z_fpath = download_data(path, url, force_download)
            subprocess.run(["uncompress", "-f", z_fpath])
            fpath = Path(z_fpath).parent.joinpath("shuttle.trn")
            df = pd.read_csv(fpath, header=None, delimiter=" ")

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
