"""Feature selection and preprocessing pipeline for team-level model inputs.

This module implements a practical, leakage-safe preprocessing flow for numeric
features built from match/team aggregates.

Pipeline strategy and intuition:
1. Correlation filtering:
    Remove near-duplicate signals first to reduce redundant features quickly.
2. VIF filtering:
    Remove features with high multicollinearity to stabilize linear models and
    improve coefficient interpretability.
3. PCA diagnostic:
    Estimate intrinsic dimensionality (95% explained variance) as a monitoring
    signal; PCA is not used to transform training inputs in this pipeline.
4. Scaling policy by feature behavior:
    - Binary indicators: impute only, no scaling (preserve 0/1 meaning).
    - Outlier-heavy numeric features: RobustScaler.
    - Remaining numeric features: StandardScaler.
5. Split-safe fitting:
    Fit preprocessing only on train, then transform val/test.
6. Output packaging:
    Return transformed train/val/test DataFrames with id and target columns
    reattached, plus a summary dict and fitted ColumnTransformer.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF scores for numeric features.

    VIF quantifies how much a feature can be explained by other features.
    Higher values indicate stronger multicollinearity and less unique signal.
    """
    if frame.shape[1] < 2:
        return pd.DataFrame({"feature": frame.columns, "vif": [1.0] * frame.shape[1]})

    numeric = frame.select_dtypes(include=["number"]).copy()
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    # NOTE: statsmodels VIF is more stable on float64 than nullable pandas dtypes.
    numeric = numeric.astype("float64").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)

    constant_cols = [
        c for c in numeric.columns if numeric[c].nunique(dropna=False) <= 1
    ]
    vif_cols = [c for c in numeric.columns if c not in constant_cols]
    if len(vif_cols) < 2:
        return pd.DataFrame(
            {"feature": numeric.columns, "vif": [1.0] * len(numeric.columns)}
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values = numeric[vif_cols].values
        vif_values = [
            variance_inflation_factor(values, i) for i in range(values.shape[1])
        ]

    vif_df = pd.DataFrame({"feature": vif_cols, "vif": vif_values})
    if constant_cols:
        vif_df = pd.concat(
            [
                vif_df,
                pd.DataFrame(
                    {"feature": constant_cols, "vif": [1.0] * len(constant_cols)}
                ),
            ],
            ignore_index=True,
        )

    return vif_df.sort_values("vif", ascending=False).reset_index(drop=True)


def _drop_high_correlation(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    threshold: float,
) -> tuple[list[str], list[str]]:
    """Drop features whose pairwise correlation exceeds the given threshold.

    Intuition: near-duplicate features usually add little new information and
    can amplify instability in downstream models.
    """
    if not feature_cols:
        return [], []

    corr_matrix = train_df[feature_cols].corr(numeric_only=True).abs()
    if corr_matrix.empty:
        return feature_cols, []

    # NOTE: We use a strict correlation cutoff to remove only near-duplicate signals.
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_drop = [col for col in upper.columns if (upper[col] > threshold).any()]
    selected = [c for c in feature_cols if c not in high_corr_drop]
    return selected, high_corr_drop


def _drop_high_vif(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    threshold: float,
    min_features: int,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Drop features with VIF above threshold, with a minimum-feature safeguard.

    Intuition: VIF filtering reduces multicollinearity, but we avoid over-pruning
    by reverting if too few features would remain.
    """
    if not feature_cols:
        return [], [], pd.DataFrame(columns=["feature", "vif"])

    vif_df = compute_vif_scores(train_df[feature_cols])
    vif_drop = vif_df.loc[vif_df["vif"] > threshold, "feature"].tolist()
    selected = [c for c in feature_cols if c not in vif_drop]

    if len(selected) < min_features:
        # NOTE: We keep coverage over aggressive pruning when VIF drops too many columns.
        selected = feature_cols
        vif_drop = []

    return selected, vif_drop, vif_df


def _pca_components_95(train_df: pd.DataFrame, feature_cols: list[str]) -> int:
    """Return number of principal components needed for 95% explained variance.

    This is a diagnostic to understand effective dimensionality, not a training
    transform in this pipeline.
    """
    if not feature_cols:
        return 0

    pca_input = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    pca_input = pca_input.fillna(pca_input.median(numeric_only=True)).fillna(0.0)

    if pca_input.shape[1] == 1:
        return 1

    scaled = StandardScaler().fit_transform(pca_input)
    pca_model = PCA(random_state=42).fit(scaled)
    cumulative = np.cumsum(pca_model.explained_variance_ratio_)
    return int(np.searchsorted(cumulative, 0.95) + 1)


def _split_scaling_groups(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Partition features into binary, robust-scaled, and standard-scaled groups.

    Intuition:
    - Binary columns keep semantic 0/1 representation.
    - Outlier-heavy columns get RobustScaler.
    - Remaining numeric columns get StandardScaler.
    """
    binary_cols: list[str] = []
    for col in feature_cols:
        series = pd.to_numeric(train_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        unique_values = set(series.unique().tolist())
        if unique_values.issubset({0.0, 1.0}):
            binary_cols.append(col)

    robust_cols: list[str] = []
    for col in feature_cols:
        if col in binary_cols:
            continue
        series = pd.to_numeric(train_df[col], errors="coerce")
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_rate = ((series < lower) | (series > upper)).mean()
        # NOTE: Robust scaling is used only for features with meaningful outlier mass (>5%).
        if outlier_rate > 0.05:
            robust_cols.append(col)

    standard_cols = [
        col for col in feature_cols if col not in robust_cols and col not in binary_cols
    ]
    return binary_cols, robust_cols, standard_cols


def fit_transform_feature_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    id_cols: list[str],
    correlation_threshold: float = 0.98,
    vif_threshold: float = 20.0,
    min_features_after_vif: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], ColumnTransformer]:
    """Select, preprocess, and transform train/val/test feature splits.

    Steps:
    1. Keep requested features that exist in train and are numeric.
    2. Apply correlation filtering.
    3. Apply VIF filtering with minimum-feature fallback.
    4. Compute PCA(95%) component count as a diagnostic.
    5. Build ColumnTransformer using binary/robust/standard groups.
    6. Fit on train only, transform val/test.
    7. Reattach id and target columns, and return summary metadata.

    Returns:
    - transformed_train, transformed_val, transformed_test: processed DataFrames
        with [id_cols + selected_features + target_col] column order.
    - summary: feature-selection/preprocessing diagnostics for reporting.
    - preprocessor: fitted ColumnTransformer for reuse in training/serving.
    """
    candidate = [col for col in feature_cols if col in train_df.columns]
    candidate_numeric = (
        train_df[candidate].select_dtypes(include=["number"]).columns.tolist()
    )

    # NOTE: Selection order is correlation -> VIF; PCA is logged as a diagnostic, not used for training.
    corr_selected, high_corr_drop = _drop_high_correlation(
        train_df,
        candidate_numeric,
        threshold=correlation_threshold,
    )
    vif_selected, vif_drop, vif_scores = _drop_high_vif(
        train_df,
        corr_selected,
        threshold=vif_threshold,
        min_features=min_features_after_vif,
    )

    pca_components = _pca_components_95(train_df, vif_selected)
    binary_cols, robust_cols, standard_cols = _split_scaling_groups(
        train_df,
        vif_selected,
    )

    transformers = []
    if binary_cols:
        # NOTE: Binary indicators are imputed but not scaled to preserve their semantics.
        binary_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        transformers.append(("binary", binary_pipeline, binary_cols))

    if robust_cols:
        robust_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )
        transformers.append(("robust", robust_pipeline, robust_cols))

    if standard_cols:
        standard_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("standard", standard_pipeline, standard_cols))

    if not transformers:
        raise ValueError("No numeric features available for preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # NOTE: Fit preprocessing on train only to avoid leaking validation/test statistics.
    x_train = preprocessor.fit_transform(train_df[vif_selected])
    x_val = preprocessor.transform(val_df[vif_selected])
    x_test = preprocessor.transform(test_df[vif_selected])

    output_feature_names = [
        name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()
    ]

    transformed_train = pd.DataFrame(
        x_train,
        columns=output_feature_names,
        index=train_df.index,
    )
    transformed_val = pd.DataFrame(
        x_val,
        columns=output_feature_names,
        index=val_df.index,
    )
    transformed_test = pd.DataFrame(
        x_test,
        columns=output_feature_names,
        index=test_df.index,
    )

    def _attach_identity_columns(
        transformed: pd.DataFrame,
        source: pd.DataFrame,
    ) -> pd.DataFrame:
        out = transformed.copy()
        for col in id_cols:
            out[col] = source[col].to_numpy()
        out[target_col] = source[target_col].to_numpy()

        ordered_cols = id_cols + output_feature_names + [target_col]
        return out[ordered_cols]

    transformed_train = _attach_identity_columns(transformed_train, train_df)
    transformed_val = _attach_identity_columns(transformed_val, val_df)
    transformed_test = _attach_identity_columns(transformed_test, test_df)

    # NOTE: Summary is the audit trail for feature pruning and preprocessing choices.
    summary: dict[str, Any] = {
        "candidate_feature_count": int(len(candidate_numeric)),
        "after_correlation": int(len(corr_selected)),
        "after_vif": int(len(vif_selected)),
        "final_selected": int(len(output_feature_names)),
        "dropped_high_correlation": high_corr_drop,
        "dropped_high_vif": vif_drop,
        "pca_components_95pct": int(pca_components),
        "binary_passthrough_cols": binary_cols,
        "robust_scaled_cols": robust_cols,
        "standard_scaled_cols": standard_cols,
        "selected_features": output_feature_names,
        "vif_top": [
            str(row.feature) for row in vif_scores.head(20).itertuples(index=False)
        ],
    }

    return transformed_train, transformed_val, transformed_test, summary, preprocessor
