"""
The code below is based on code from the SforAiDl/GenRL library https://github.com/SforAiDl/genrl
distributed under the following license:

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
import random
from typing import List, Tuple, Union
import torch


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


class DataBasedBandit(Bandit):
    """Base class for contextual bandits based on  datasets.
    Args:
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".
    Attributes:
        device (torch.device): Device to use for tensor operations.
    """

    def __init__(self, device: str = "cpu", **kwargs):

        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self._reset()

    @property
    def reward_hist(self) -> List[float]:
        """List[float]: History of rewards generated."""
        return self._reward_hist

    @property
    def regret_hist(self) -> List[float]:
        """List[float]: History of regrets generated."""
        return self._regret_hist

    @property
    def cum_reward_hist(self) -> Union[List[int], List[float]]:
        """List[float]: History of cumulative rewards generated."""
        return self._cum_regret_hist

    @property
    def cum_regret_hist(self) -> Union[List[int], List[float]]:
        """List[float]: History of cumulative regrets generated."""
        return self._cum_reward_hist

    @property
    def cum_regret(self) -> Union[int, float]:
        """Union[int, float]: Cumulative regret."""
        return self._cum_regret

    @property
    def cum_reward(self) -> Union[int, float]:
        """Union[int, float]: Cumulative reward."""
        return self._cum_reward

    def _reset(self):
        """Resets tracking metrics."""
        self.idx = 0
        self._cum_regret = 0
        self._cum_reward = 0
        self._reward_hist = []
        self._regret_hist = []
        self._cum_regret_hist = []
        self._cum_reward_hist = []

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        """Compute the reward for a given action.
        Note:
            This method needs to be implemented in the specific bandit.
        Args:
            action (int): The action to compute reward for.
        Returns:
            Tuple[int, int]: Computed reward.
        """
        raise NotImplementedError

    def _get_context(self) -> torch.Tensor:
        """Get the vector for current selected context.
        Note:
            This method needs to be implemented in the specific bandit.
        Returns:
            torch.Tensor: Current context vector.
        """
        raise NotImplementedError

    def step(self, action: int) -> Tuple[torch.Tensor, int]:
        """Generate reward for given action and select next context.
        This method also updates the various regret and reward trackers
        as well the current context index.
        Args:
            action (int): Selected action.
        Returns:
            Tuple[torch.Tensor, int]: Tuple of the next context and the
                reward generated for given action
        """
        reward, max_reward = self._compute_reward(action)
        regret = max_reward - reward
        self._cum_regret += regret
        self.cum_regret_hist.append(self._cum_regret)
        self.regret_hist.append(regret)
        self._cum_reward += reward
        self.cum_reward_hist.append(self._cum_reward)
        self.reward_hist.append(reward)
        self.idx += 1
        if not self.idx < self.len:
            self.idx = 0
        context = self._get_context()
        return context, reward

    def reset(self) -> torch.Tensor:
        """Reset bandit by shuffling indices and get new context.
        Note:
            This method needs to be implemented in the specific bandit.
        Returns:
            torch.Tensor: Current context selected by bandit.
        """
        raise NotImplementedError


class TransitionDB(object):
    """
    Database for storing (context, action, reward) transitions.
    Args:
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".
    Attributes:
        db (dict): Dictionary containing list of transitions.
        db_size (int): Number of transitions stored in database.
        device (torch.device): Device to use for tensor operations.
    """

    def __init__(self, device: Union[str, torch.device] = "cpu"):

        if type(device) is str:
            self.device = (
                torch.device(device)
                if "cuda" in device and torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device

        self.db = {"contexts": [], "actions": [], "rewards": []}
        self.db_size = 0

    def add(self, context: torch.Tensor, action: int, reward: int):
        """Add (context, action, reward) transition to database
        Args:
            context (torch.Tensor): Context recieved
            action (int): Action taken
            reward (int): Reward recieved
        """
        self.db["contexts"].append(context)
        self.db["actions"].append(action)
        self.db["rewards"].append(reward)
        self.db_size += 1

    def get_data(
        self, batch_size: Union[int, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a batch of transition from database
        Args:
            batch_size (Union[int, None], optional): Size of batch required.
                Defaults to None which implies all transitions in the database
                are to be included in batch.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of stacked
                contexts, actions, rewards tensors.
        """
        if batch_size is None:
            batch_size = self.db_size
        else:
            batch_size = min(batch_size, self.db_size)
        idx = [random.randrange(self.db_size) for _ in range(batch_size)]
        x = (
            torch.stack([self.db["contexts"][i] for i in idx])
            .to(self.device)
            .to(torch.float)
        )
        y = (
            torch.tensor([self.db["rewards"][i] for i in idx])
            .to(self.device)
            .to(torch.float)
            .unsqueeze(1)
        )
        a = (
            torch.stack([self.db["actions"][i] for i in idx])
            .to(self.device)
            .to(torch.long)
        )
        return x, a, y

    def get_data_for_action(
        self, action: int, batch_size: Union[int, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of transition from database for a given action.
        Args:
            action (int): The action to sample transitions for.
            batch_size (Union[int, None], optional): Size of batch required.
                Defaults to None which implies all transitions in the database
                are to be included in batch.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of stacked
                contexts and rewards tensors.
        """
        action_idx = [i for i in range(self.db_size) if self.db["actions"][i] == action]
        if batch_size is None:
            t_batch_size = len(action_idx)
        else:
            t_batch_size = min(batch_size, len(action_idx))
        idx = random.sample(action_idx, t_batch_size)
        x = (
            torch.stack([self.db["contexts"][i] for i in idx])
            .to(self.device)
            .to(torch.float)
        )
        y = (
            torch.tensor([self.db["rewards"][i] for i in idx])
            .to(self.device)
            .to(torch.float)
            .unsqueeze(1)
        )
        return x, y

