from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor, from_numpy

from dataset2vec.utils import (
    DataUtils,
    InconsistentTypesException,
    InvalidDataTypeException,
)


class Dataset2VecLoader:
    """
    Dataloader responsible for the generation of the
    examples needed for the training of the Dataset2Vec.
    In each iteration it returns tuple :math:`(X_1, y_1, X_2, y_2, label)`.
    :math:`X_1, X_2` are subsets (both in terms of records and columns) of
    the features matrices of the passed datasets. :math:`y_1, y_2` are subsets
    of the targets of the datasets (as for now it is the last column of
    the dataset). Label is equal to 1 when :math:`(X_1, y_1)`
    and :math:`(X_2, y_2)` originate from the same dataset and 0 otherwise.
    :math:`X_1, y_1, X_2, y_2` are ``torch.Tensor``.

    Args:
        data (Path | list[Path] | list[pd.DataFrame] | list[NDArray] | list[Tensor]):
            input data to the loader. If Path, then all
            csv files are read from this directory. If the list of
            paths then csv files under these paths are read. If
            pd.DataFrame or np.NDArray, then they are converted
            to torch.Tensor. During the creation of the loader
            the data is imputed, standardized and categorical
            columns are one-hot encoded

        batch_size (int, optional): Number of the observations
            in the single batch. Defaults to 32.

        n_batches (int, optional): Number of batches that loader
            can generate. Defaults to 100.
    """  # noqa: E501

    def __init__(
        self,
        data: (
            Path
            | list[Path]
            | list[pd.DataFrame]
            | list[NDArray[np.generic]]
            | list[Tensor]
        ),
        batch_size: int = 32,
        n_batches: int = 100,
    ):
        self.data = data
        self.batch_size = batch_size
        self.n_batches = n_batches

        datasets = self.__read_data_if_needed(data)
        self.n_datasets = len(datasets)
        self.released_batches_count = 0

        self.__setup_xs(datasets)
        self.__setup_ys(datasets)

    def __setup_xs(
        self,
        datasets: (
            list[pd.DataFrame] | list[NDArray[np.generic]] | list[Tensor]
        ),
    ) -> None:
        Xs = [
            self.__normalize_type_to_pandas(dataset).iloc[:, :-1]
            for dataset in datasets
        ]
        Xs = [
            DataUtils.get_preprocessing_pipeline().fit_transform(X).values
            for X in Xs
        ]
        self.Xs = [self.__convert_numpy_to_torch(X) for X in Xs]

    def __setup_ys(
        self,
        datasets: (
            list[pd.DataFrame] | list[NDArray[np.generic]] | list[Tensor]
        ),
    ) -> None:
        ys = [
            self.__normalize_type_to_pandas(dataset).iloc[:, -1]
            for dataset in datasets
        ]
        self.ys = [
            self.__convert_numpy_to_torch(y.values).reshape(-1, 1) for y in ys
        ]

    def __read_data_if_needed(
        self,
        data: (
            Path
            | list[Path]
            | list[pd.DataFrame]
            | list[NDArray[np.generic]]
            | list[Tensor]
        ),
    ) -> list[pd.DataFrame] | list[NDArray[np.generic]] | list[Tensor]:
        if isinstance(data, Path):
            return [pd.read_csv(file) for file in sorted(data.iterdir())]
        elif (
            isinstance(data, list)
            and len(data) > 0
            and isinstance(data[0], Path)
        ):
            if any(map(lambda el: not isinstance(el, Path), data)):
                raise InconsistentTypesException(
                    "If any element of the list is Path"
                    "then all elements should be Path"
                )
            return [pd.read_csv(file) for file in data]
        else:
            return data

    def __convert_numpy_to_torch(self, data: NDArray[np.generic]) -> Tensor:
        if isinstance(data, np.ndarray):
            data_converted = from_numpy(data).type(torch.float)
            return data_converted
        else:
            raise InvalidDataTypeException(
                f"{type(data)} is not a NDArray type."
            )

    def __normalize_type_to_pandas(
        self, data: pd.DataFrame | Tensor | NDArray[np.generic]
    ) -> pd.DataFrame:
        if isinstance(data, Tensor):
            return pd.DataFrame(data.numpy())
        elif isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        else:
            raise InvalidDataTypeException(
                f"{type(data)} is not supported by the"
                "Dataset2VecLoader. Loader supports torch.Tensor,"
                "pandas.DataFrame or NDArray."
            )

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Dataset2VecLoader:
        return deepcopy(self)

    def __next__(self) -> list[tuple[Tensor, Tensor, Tensor, Tensor, int]]:
        if self.released_batches_count == self.n_batches:
            raise StopIteration()
        self.released_batches_count += 1
        return self.__get_batch()

    def __get_batch(
        self,
    ) -> list[tuple[Tensor, Tensor, Tensor, Tensor, int]]:
        return [self.__get_single_example() for _ in range(self.batch_size)]

    def __get_single_example(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        dataset_1_idx, dataset_2_idx = self.__get_random_datasets_indices()
        return (
            *self.__generate_dataset_subsample(dataset_1_idx),
            *self.__generate_dataset_subsample(dataset_2_idx),
            int(dataset_1_idx == dataset_2_idx),
        )

    def __get_random_datasets_indices(
        self,
    ) -> tuple[int, int]:
        if np.random.uniform() >= 0.5:
            idx = np.random.choice(self.n_datasets, 1)[0]
            return (idx, idx)
        else:
            idx1, idx2 = np.random.choice(
                self.n_datasets, 2, replace=False
            ).astype(int)
            return (idx1, idx2)

    def __generate_dataset_subsample(
        self, dataset_idx: int
    ) -> tuple[Tensor, Tensor]:
        X, y = self.Xs[dataset_idx], self.ys[dataset_idx]
        rows_idx, features_idx, targets_idx = self.__sample_batch_idx(X, y)
        X = DataUtils.index_tensor_using_lists(X, rows_idx, features_idx)
        y = DataUtils.index_tensor_using_lists(y, rows_idx, targets_idx)
        return X, y

    def __sample_batch_idx(
        self, X: Tensor, y: Tensor
    ) -> tuple[NDArray[np.generic], NDArray[np.generic], NDArray[np.generic]]:
        n_rows = X.shape[0]
        assert n_rows >= 8

        n_features = X.shape[1]
        n_targets = y.shape[1]

        max_q = min(int(np.log2(n_rows)), 8)
        q = np.random.choice(np.arange(3, max_q + 1))
        n_rows_to_select = 2**q
        rows_idx = np.random.choice(n_rows, n_rows_to_select)
        features_idx = DataUtils.sample_random_subset(n_features, max_samples=128)
        targets_idx = DataUtils.sample_random_subset(n_targets)

        return rows_idx, features_idx, targets_idx


class RepeatableDataset2VecLoader:
    """
    Loader with similar interface to Dataset2VecLoader but it
    returns on each iter call the same list of batches. Useful for the
    validation and testing purposes.

    Args:
        data (Path | list[Path] | list[pd.DataFrame] | list[NDArray] | list[Tensor]):
            input data to the loader. If Path, then all
            csv files are read from this directory. If the list of
            paths then csv files under these paths are read. If
            pd.DataFrame or np.NDArray, then they are converted
            to torch.Tensor. During the creation of the loader
            the data is imputed, standardized and categorical
            columns are one-hot encoded

        batch_size (int, optional): Number of the observations
            in the single batch. Defaults to 32.

        n_batches (int, optional): Number of batches that loader
            can generate. Defaults to 100.
    """  # noqa: E501

    def __init__(
        self,
        data: (
            Path
            | list[Path]
            | list[pd.DataFrame]
            | list[NDArray[np.generic]]
            | list[Tensor]
        ),
        batch_size: int = 32,
        n_batches: int = 100,
    ):
        loader = Dataset2VecLoader(data, batch_size, n_batches)
        self.batches = list(loader)
        self.released_batches_count = 0

    def __next__(self) -> list[tuple[Tensor, Tensor, Tensor, Tensor, int]]:
        if self.released_batches_count == len(self.batches):
            raise StopIteration()
        batch = self.batches[self.released_batches_count]
        self.released_batches_count += 1
        return batch

    def __iter__(self) -> RepeatableDataset2VecLoader:
        return deepcopy(self)
