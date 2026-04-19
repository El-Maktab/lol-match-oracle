from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.constants import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE


@dataclass(slots=True)
class DatasetSplit:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def split_train_val_test(
    df: pd.DataFrame,
    *,
    target_col: str = "win",
    group_col: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DatasetSplit:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in dataframe")

    y = df[target_col]

    if group_col is None:
        x = df.drop(columns=[target_col])

        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        adjusted_val_size = val_size / (1 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=y_train_val,
        )
    else:
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' was not found in dataframe")

        x = df.drop(columns=[target_col])
        groups = df[group_col]
        unique_groups = pd.Index(groups.dropna().unique())

        train_val_groups, test_groups = train_test_split(
            unique_groups,
            test_size=test_size,
            random_state=random_state,
        )

        adjusted_val_size = val_size / (1 - test_size)
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=adjusted_val_size,
            random_state=random_state,
        )

        train_mask = groups.isin(train_groups)
        val_mask = groups.isin(val_groups)
        test_mask = groups.isin(test_groups)

        if set(train_groups).intersection(set(val_groups)):
            raise ValueError("Train and validation groups overlap.")
        if set(train_groups).intersection(set(test_groups)):
            raise ValueError("Train and test groups overlap.")
        if set(val_groups).intersection(set(test_groups)):
            raise ValueError("Validation and test groups overlap.")

        x_train = x.loc[train_mask]
        x_val = x.loc[val_mask]
        x_test = x.loc[test_mask]
        y_train = y.loc[train_mask]
        y_val = y.loc[val_mask]
        y_test = y.loc[test_mask]

    return DatasetSplit(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
