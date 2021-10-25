import copy
import os
import sys
import time
from collections import MutableMapping
from random import randint
from typing import Any, Dict, List

import torch


class ddict(object):
    '''
    dd = ddict(lr=[0.1, 0.2], n_hiddens=[100, 500, 1000], n_layers=2)
    dd._extend(['lr', 'n_hiddens'], [[0.3, 0.4], [2000]])
    # Save to file:
    dd._save('my_file', date=False)
    # Load ddict from file:
    new_dd = ddict()._load('my_file')
    '''

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __add__(self, other):
        if isinstance(other, type(self)):
            sum_dct = copy.copy(self.__dict__)
            for k, v in other.__dict__.items():
                if k not in sum_dct:
                    sum_dct[k] = v
                else:
                    if type(v) is list and type(sum_dct[k]) is list:
                        sum_dct[k] = sum_dct[k] + v
                    elif type(v) is not list and type(sum_dct[k]) is list:
                        sum_dct[k] = sum_dct[k] + [v]
                    elif type(v) is list and type(sum_dct[k]) is not list:
                        sum_dct[k] = [sum_dct[k]] + v
                    else:
                        sum_dct[k] = [sum_dct[k]] + [v]
            return ddict(**sum_dct)

        elif isinstance(other, dict):
            return self.__add__(ddict(**other))
        else:
            raise ValueError("ddict or dict is required")

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in self._keys())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    @staticmethod
    def _flatten_dict(d, parent_key='', sep='_'):
        "Recursively flattens nested dicts"
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(ddict._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _extend(self, keys, values_list):
        if type(keys) not in (tuple, list):  # Individual key
            if keys not in self._keys():
                self[keys] = values_list
            else:
                self[keys] += values_list
        else:
            for key, val in zip(keys, values_list):
                if type(val) is list:
                    self._extend(key, val)
                else:
                    self._extend(key, [val])
        return self

    def _keys(self):
        return tuple(sorted([k for k in self.__dict__ if not k.startswith('_')]))

    def _values(self):
        return tuple([self.__dict__[k] for k in self._keys()])

    def _items(self):
        return tuple(zip(self._keys(), self._values()))

    def _save(self, filename=None, date=True):
        if filename is None:
            if not hasattr(self, '_filename'):  # First save
                raise ValueError("filename must be provided the first time you call _save()")
            else:  # Already saved
                torch.save(self, self._filename + '.pt')
        else:  # New filename
            if date:
                filename += '_' + time.strftime("%Y%m%d-%H:%M:%S")
            # Check that directory exists, otherwise create it
            dir_name = os.path.dirname(filename)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            # Check if filename does not already exist. If it does, change name.
            while os.path.exists(filename + '.pt') and len(filename) < 100:
                filename += str(randint(0, 9))
            self._filename = filename
            torch.save(self, self._filename + '.pt')
        return self

    def _load(self, filename):
        try:
            self = torch.load(filename)
        except FileNotFoundError:
            self = torch.load(filename + '.pt')
        return self

    def _to_dict(self):
        "Returns a dict (it's recursive)"
        return_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, type(self)):
                return_dict[k] = v._to_dict()
            else:
                return_dict[k] = v
        return return_dict

    def _flatten(self, parent_key='', sep='_'):
        "Recursively flattens nested ddicts"
        d = self._to_dict()
        return ddict._flatten_dict(d)


def show_progress(amount_so_far, total_amount):
    r"""Shows progress along a process in one stdout line
    Examples::
        >>> import time
        >>> for i in range(30):
        >>>     show_progress(i+1, 30)
        >>>     time.sleep(0.2)
    """
    percent = float(amount_so_far) / total_amount
    percent = round(percent * 100, 2)
    sys.stdout.write("  Progress %d of %d (%0.2f%%)\r" % (amount_so_far, total_amount, percent))
    if amount_so_far >= total_amount:
        sys.stdout.write('\n')


def get_device(seed=1):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.manual_seed(seed)
    # Multi GPU?
    num_gpus = torch.cuda.device_count()
    if device.type != 'cpu':
        print('\033[93m' + 'Using CUDA,', num_gpus, 'GPUs\033[0m')
        torch.cuda.manual_seed(seed)
    return device


class Logger:
    """
    Logger class to log important information
    :param logdir: Directory to save log at
    :param formats: Formatting of each log ['csv', 'stdout', 'tensorboard']
    :type logdir: string
    :type formats: list
    """

    def __init__(self, logdir: str = None, formats: List[str] = ["csv"]):
        if logdir is None:
            self._logdir = os.getcwd()
        else:
            self._logdir = logdir
            if not os.path.isdir(self._logdir):
                os.makedirs(self._logdir)
        self._formats = formats
        self.writers = []
        for ft in self.formats:
            self.writers.append(get_logger_by_name(ft)(self.logdir))

    def write(self, kvs: Dict[str, Any], log_key: str = "timestep") -> None:
        """
        Add entry to logger
        :param kvs: Entry to be logged
        :param log_key: Key plotted on log_key
        :type kvs: dict
        :type log_key: str
        """
        for writer in self.writers:
            writer.write(kvs, log_key)

    def close(self) -> None:
        """
        Close the logger
        """
        for writer in self.writers:
            writer.close()

    @property
    def logdir(self) -> str:
        """
        Return log directory
        """
        return self._logdir

    @property
    def formats(self) -> List[str]:
        """
        Return save format(s)
        """
        return self._formats


class HumanOutputFormat:
    """
    Output from a log file in a human readable format
    :param logdir: Directory at which log is present
    :type logdir: string
    """

    def __init__(self, logdir: str):
        self.file = os.path.join(logdir, "train.log")
        self.first = True
        self.lens = []
        self.maxlen = 0

    def write(self, kvs: Dict[str, Any], log_key) -> None:
        """
        Log the entry out in human readable format
        :param kvs: Entries to be logged
        :type kvs: dict
        """
        self.write_to_file(kvs, sys.stdout)
        with open(self.file, "a") as file:
            self.write_to_file(kvs, file)

    def write_to_file(self, kvs: Dict[str, Any], file=sys.stdout) -> None:
        """
        Log the entry out in human readable format
        :param kvs: Entries to be logged
        :param file: Name of file to write logs to
        :type kvs: dict
        :type file: io.TextIOWrapper
        """
        if self.first:
            self.first = False
            self.max_key_len(kvs)
            for key, value in kvs.items():
                print(
                    "{}{}".format(str(key), " " * (self.maxlen - len(str(key)))),
                    end="  ",
                    file=file,
                )
            print()
        for key, value in kvs.items():
            rounded = self.round(value)
            print(
                "{}{}".format(rounded, " " * (self.maxlen - len(str(rounded)))),
                end="  ",
                file=file,
            )
        print("", file=file)

    def max_key_len(self, kvs: Dict[str, Any]) -> None:
        """
        Finds max key length
        :param kvs: Entries to be logged
        :type kvs: dict
        """
        self.lens = [len(str(key)) for key, value in kvs.items()]
        maxlen = max(self.lens)
        self.maxlen = maxlen
        if maxlen < 15:
            self.maxlen = 15

    def round(self, num: float) -> float:
        """
        Returns a rounded float value depending on self.maxlen
        :param num: Value to round
        :type num: float
        """
        exponent_len = len(str(num // 1.0)[:-2])
        rounding_len = min(self.maxlen - exponent_len, 4)
        return round(num, rounding_len)

    def close(self) -> None:
        pass


class CSVLogger:
    """
    CSV Logging class
    :param logdir: Directory to save log at
    :type logdir: string
    """

    def __init__(self, logdir: str):
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)
        self.file = open("{}/train.csv".format(logdir), "w")
        self.first = True
        self.keynames = {}

    def write(self, kvs: Dict[str, Any], log_key) -> None:
        """
        Add entry to logger
        :param kvs: Entries to be logged
        :type kvs: dict
        """
        if self.first:
            for i, key in enumerate(kvs.keys()):
                self.keynames[key] = i
                self.file.write(key)
                self.file.write(",")
            self.file.write("\n")
            self.first = False

        for i, (key, value) in enumerate(kvs.items()):
            if key not in self.keynames.keys():
                raise Exception(
                    "A new value '{}' cannot be added to CSVLogger".format(key)
                )
            if i != self.keynames[key]:
                raise Exception("Value not at the same index as when initialized")
            self.file.write(str(value))
            self.file.write(",")

        self.file.write("\n")

    def close(self) -> None:
        """
        Close the logger
        """
        self.file.close()


logger_registry = {
    "stdout": HumanOutputFormat,
    "csv": CSVLogger,
}


def get_logger_by_name(name: str):
    """
    Gets the logger given the type of logger
    :param name: Name of the value function needed
    :type name: string
    :returns: Logger
    """
    if name not in logger_registry.keys():
        raise NotImplementedError
    else:
        return logger_registry[name]
