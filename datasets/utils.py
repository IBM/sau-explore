import gzip
import shutil
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd


def safe_std(data):
    """Remove zero std values for ones."""
    std = np.std(data, axis=0)
    return np.array([val if val != 0.0 else 1.0 for val in std])


def safe_zscore(data):
    mean = np.mean(data, axis=0, keepdims=True)
    return (data - mean) / safe_std(data)


def normalize_cat_num(df, categorical_columns):
    """Normalizes (z-scores) numerical columns and puts categorical variables
        in one-hot format

    Inputs:
        df (Pandas dataframe): dataset
        categorical_columns (list): list of categorical columns

    Outputs:

    """
    n_numerical = df.shape[1] - len(categorical_columns)
    df = pd.get_dummies(df, columns=categorical_columns)

    # First n_numerical columns are numerical
    X_num = safe_zscore(df.iloc[:, :n_numerical].to_numpy())
    # Remaining columns are categorical
    X_cat = df.iloc[:, n_numerical:].to_numpy()
    return np.concatenate((X_num, X_cat), axis=1)


def get_values(pd_series):
    values = pd_series.astype('category').cat.codes.to_numpy()
    n_values = len(np.unique(values))
    return values, n_values


def download_data(path, url, force=False, filename=None):
    """Download data to given location from given URL.
    Args:
        path (str): Location to download to.
        url (str): URL to download file from.
        force (bool, optional): Force download even if file exists. Defaults to False.
        filename (Union[str, None], optional): Name to save file under. Defaults to None
            which implies original filename is to be used.
    Returns:
        str: Path to downloaded file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = Path(url).name
    fpath = path.joinpath(filename)
    if fpath.is_file() and not force:
        return str(fpath)

    try_data_download(fpath, path, url)

    return str(fpath)


def try_data_download(fpath, path, url):
    try:
        print(f"Downloading {url} to {fpath.resolve()}")
        urllib.request.urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Failed download. Trying https -> http instead.")
            print(f" Downloading {url} to {path}")
            urllib.request.urlretrieve(url, fpath)
        else:
            raise e


def fetch_data_without_header(path, fname, delimiter=",", na_values=None):
    if na_values is None:
        na_values = []
    if Path(path).is_dir():
        path = Path(path).joinpath(fname)
    if Path(path).is_file():
        df = pd.read_csv(
            path, header=None, delimiter=delimiter, na_values=na_values
        ).dropna()
    else:
        raise FileNotFoundError(f"File not found at location {path}, use download flag")
    return df


def fetch_data_with_header(path, fname, delimiter=",", na_values=None):
    if na_values is None:
        na_values = []
    if Path(path).is_dir():
        path = Path(path).joinpath(fname)
    if Path(path).is_file():
        df = pd.read_csv(path, delimiter=delimiter, na_values=na_values).dropna()
    else:
        raise FileNotFoundError(f"File not found at location {path}, use download flag")
    return df


def fetch_zipped_data_without_header(gz_fpath, delimiter=",", na_values=None):
    if na_values is None:
        na_values = []
    with gzip.open(gz_fpath, "rb") as f_in:
        fpath = Path(gz_fpath).parent.joinpath("covtype.data")
        with open(fpath, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    df = pd.read_csv(
        fpath, header=None, delimiter=delimiter, na_values=na_values
    ).dropna()
    fpath.unlink()
    return df
