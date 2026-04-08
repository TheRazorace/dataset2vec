from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch import Tensor


class DataUtils:

    @staticmethod
    def get_preprocessing_pipeline() -> BaseEstimator:
        """
        Returns sklearn preprocessing pipeline for the data which:
            * Imputes and standardizes numerical columns
            * Imputes and one-hot encodes categorical columns

        Returns:
            BaseEstimator: preprocessing pipeline in sklearn
                format
        """
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "one-hot",
                    OneHotEncoder(
                        sparse_output=False, handle_unknown="ignore"
                    ),
                ),
            ]
        ).set_output(transform="pandas")

        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", MinMaxScaler()),
            ]
        ).set_output(transform="pandas")

        pipeline = Pipeline(
            [
                (
                    "transformers",
                    make_column_transformer(
                        (
                            cat_pipeline,
                            make_column_selector(
                                dtype_include=("object", "category")
                            ),
                        ),
                        (
                            num_pipeline,
                            make_column_selector(dtype_include=np.number),
                        ),
                    ),
                )
            ]
        ).set_output(transform="pandas")
        return pipeline

    @staticmethod
    def sample_random_subset(
        a: int | NDArray[np.generic],
        max_samples: int | None = None,
    ) -> NDArray[np.generic]:
        """
        Samples random subset with variable length of the given array. When int
        is passed then samples from the range [0, a - 1]. Subsets are sampled
        from the uniform distribution.

        Args:
            a (int | NDArray[np.generic]): Array to sample from.
                When int is passed the samples from the range [0, a - 1].
            max_samples (int | None): Optional ceiling on the maximum number
                of items included. Subsetting will not exceed this.

        Returns:
            NDArray[np.generic]: Subsample of the input array.
        """
        if isinstance(a, int):
            a = np.arange(a)

        if len(a) == 1:
            return a
        subset_idx = np.random.uniform(size=len(a)) < 0.5
        if np.sum(subset_idx) == 0:
            subset_idx[np.random.choice(len(a))] = True  # Fallback to at least 1 unconditionally
            
        sampled = a[subset_idx]
        
        # Ceil the sampled elements if a limit is mandated
        if max_samples is not None and len(sampled) > max_samples:
            sampled = np.random.choice(sampled, max_samples, replace=False)
            
        return sampled

    @staticmethod
    def index_tensor_using_lists(
        tensor: Tensor,
        rows_idx: NDArray[np.generic],
        col_idx: NDArray[np.generic],
    ) -> Tensor:
        """
        Returns subarray of the input tensor from rows and columns
        from given indices.

        Args:
            tensor (Tensor): Tensor to index.

            rows_idx (NDArray[np.generic]): Index of the rows.

            col_idx (NDArray[np.generic]): Index of the columns.

        Returns:
            Tensor: Tensor with specified rows and columns.
        """
        return tensor[rows_idx].T[col_idx].T


class Validators:

    @staticmethod
    def is_positive(number: int) -> int:
        assert number > 0, "Number is non-positive"
        return number

    @staticmethod
    def non_negative(number: int) -> int:
        assert number >= 0, "Number is negative"
        return number

    @staticmethod
    def all_elements_positive(arr: list[int]) -> list[int]:
        assert all(
            map(lambda x: x > 0, arr)
        ), "List contains non-positive elements"
        return arr

    @staticmethod
    def non_empty(arr: list[int]) -> list[int]:
        assert len(arr) > 0, "List is empty"
        return arr


class InvalidDataTypeException(Exception):
    pass


class InconsistentTypesException(Exception):
    pass
