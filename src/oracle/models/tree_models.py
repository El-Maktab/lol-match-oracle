from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

TREE_MODEL_NAMES = (
    "random_forest",
    "xgboost",
    "lightgbm",
)


class TreeClassifier(Protocol):
    """Interface contract for tree-based classifiers used by the Trainer."""

    def fit(self, x: Any, y: Any) -> Any:
        """Fit model parameters on the provided training matrix."""

    def predict(self, x: Any) -> Any:
        """Predict class labels for the provided matrix."""


@dataclass(slots=True)
class RandomForestTreeClassifier:
    """Random-forest wrapper with deterministic defaults and sklearn parity."""

    n_estimators: int
    max_depth: int | None
    min_samples_split: int
    min_samples_leaf: int
    class_weight: str | dict[int, float] | None
    n_jobs: int | None
    random_state: int
    _model: RandomForestClassifier = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        x_val: Any | None = None,
        y_val: Any | None = None,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> RandomForestTreeClassifier:
        """Fit random forest; validation split is accepted for API uniformity."""

        del x_val
        del y_val
        del show_progress
        del progress_desc

        self._model.fit(x, y)
        return self

    def predict(self, x: Any) -> np.ndarray:
        """Predict binary class labels."""

        return np.asarray(self._model.predict(x), dtype=int)

    def predict_proba(self, x: Any) -> np.ndarray:
        """Return class probabilities for ROC-AUC and calibration diagnostics."""

        return np.asarray(self._model.predict_proba(x), dtype=float)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Expose feature importances for trainer artifact logging."""

        return np.asarray(self._model.feature_importances_, dtype=float)


@dataclass(slots=True)
class XGBoostTreeClassifier:
    """XGBoost wrapper with optional validation-based early stopping."""

    n_estimators: int
    learning_rate: float
    max_depth: int
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    min_child_weight: float
    gamma: float
    early_stopping_rounds: int
    eval_metric: str
    n_jobs: int | None
    random_state: int
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if importlib.util.find_spec("xgboost") is None:
            raise ImportError(
                "xgboost is not installed. Install xgboost to enable 'xgboost' model."
            )

        xgboost_module = importlib.import_module("xgboost")
        xgb_classifier = getattr(xgboost_module, "XGBClassifier")

        self._model = xgb_classifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        x_val: Any | None = None,
        y_val: Any | None = None,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> XGBoostTreeClassifier:
        """Fit XGBoost with early stopping when validation data is provided."""

        xgboost_module = importlib.import_module("xgboost")
        callback_module = getattr(xgboost_module, "callback", None)

        fit_kwargs: dict[str, Any] = {}
        callbacks: list[Any] = []
        if x_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(x_val, y_val)]
            fit_kwargs["verbose"] = False
            if self.early_stopping_rounds > 0:
                if callback_module is not None and hasattr(
                    callback_module, "EarlyStopping"
                ):
                    callbacks.append(
                        callback_module.EarlyStopping(
                            rounds=self.early_stopping_rounds,
                            save_best=True,
                        )
                    )
                else:
                    fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        if show_progress:
            if callback_module is None or not hasattr(
                callback_module, "TrainingCallback"
            ):
                raise RuntimeError(
                    "xgboost callback API is unavailable for progress tracking."
                )

            class XGBoostTqdmCallback(callback_module.TrainingCallback):
                def __init__(self, *, total: int, desc: str) -> None:
                    self.total = max(1, total)
                    self.desc = desc
                    self._progress: Any | None = None

                def before_training(self, model: Any) -> Any:
                    self._progress = tqdm(
                        total=self.total,
                        desc=self.desc,
                        unit="iter",
                        leave=False,
                    )
                    return model

                def after_iteration(
                    self,
                    model: Any,
                    epoch: int,
                    evals_log: Any,
                ) -> bool:
                    del model
                    del epoch
                    del evals_log
                    if self._progress is not None:
                        self._progress.update(1)
                    return False

                def after_training(self, model: Any) -> Any:
                    if self._progress is not None:
                        self._progress.close()
                    return model

            callbacks = list(fit_kwargs.get("callbacks", []))
            callbacks.append(
                XGBoostTqdmCallback(
                    total=self.n_estimators,
                    desc=progress_desc or "xgboost fit",
                )
            )
            fit_kwargs["verbose"] = False

        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        try:
            self._model.fit(x, y, **fit_kwargs)
        except TypeError:
            # NOTE: Handle older/newer xgboost sklearn APIs without crashing training.
            fallback_kwargs = dict(fit_kwargs)
            fallback_kwargs.pop("callbacks", None)
            fallback_kwargs.pop("early_stopping_rounds", None)
            self._model.fit(x, y, **fallback_kwargs)
        return self

    def predict(self, x: Any) -> np.ndarray:
        """Predict binary class labels."""

        return np.asarray(self._model.predict(x), dtype=int)

    def predict_proba(self, x: Any) -> np.ndarray:
        """Return class probabilities for downstream metrics."""

        return np.asarray(self._model.predict_proba(x), dtype=float)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Expose feature importances for trainer artifact logging."""

        return np.asarray(self._model.feature_importances_, dtype=float)


@dataclass(slots=True)
class LightGBMTreeClassifier:
    """LightGBM wrapper with optional validation-based early stopping."""

    n_estimators: int
    learning_rate: float
    num_leaves: int
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    min_child_samples: int
    class_weight: str | dict[int, float] | None
    early_stopping_rounds: int
    n_jobs: int | None
    random_state: int
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if importlib.util.find_spec("lightgbm") is None:
            raise ImportError(
                "lightgbm is not installed. Install lightgbm to enable 'lightgbm' model."
            )

        lightgbm_module = importlib.import_module("lightgbm")
        lgbm_classifier = getattr(lightgbm_module, "LGBMClassifier")

        self._model = lgbm_classifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_samples=self.min_child_samples,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        x_val: Any | None = None,
        y_val: Any | None = None,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> LightGBMTreeClassifier:
        """Fit LightGBM with early stopping when validation data is provided."""

        fit_kwargs: dict[str, Any] = {}
        callbacks: list[Any] = []
        if x_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(x_val, y_val)]
            if self.early_stopping_rounds > 0:
                lightgbm_module = importlib.import_module("lightgbm")
                callbacks.append(
                    lightgbm_module.early_stopping(
                        self.early_stopping_rounds,
                        verbose=False,
                    )
                )

        progress: Any | None = None
        if show_progress:
            progress = tqdm(
                total=max(1, self.n_estimators),
                desc=progress_desc or "lightgbm fit",
                unit="iter",
                leave=False,
            )
            last_iteration = {"value": 0}

            def _lightgbm_progress_callback(env: Any) -> None:
                current_iteration = int(env.iteration) + 1
                delta = max(0, current_iteration - last_iteration["value"])
                if progress is not None and delta > 0:
                    progress.update(delta)
                last_iteration["value"] = current_iteration

            callbacks.append(_lightgbm_progress_callback)

        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        try:
            self._model.fit(x, y, **fit_kwargs)
        finally:
            if progress is not None:
                progress.close()
        return self

    def predict(self, x: Any) -> np.ndarray:
        """Predict binary class labels."""

        return np.asarray(self._model.predict(x), dtype=int)

    def predict_proba(self, x: Any) -> np.ndarray:
        """Return class probabilities for downstream metrics."""

        return np.asarray(self._model.predict_proba(x), dtype=float)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Expose feature importances for trainer artifact logging."""

        return np.asarray(self._model.feature_importances_, dtype=float)


def _normalize_class_weight(value: Any) -> str | dict[int, float] | None:
    if value is None:
        return None

    if isinstance(value, str) and value.lower() in {"none", "null", "~", ""}:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        normalized: dict[int, float] = {}
        for key, weight in value.items():
            normalized[int(key)] = float(weight)
        return normalized

    raise TypeError("Unsupported class_weight value for tree model.")


def _normalize_optional_int(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, str) and value.strip().lower() in {"none", "null", "~", ""}:
        return None

    return int(value)


def build_tree_model(
    model_name: str,
    *,
    params: Mapping[str, Any],
    random_state: int,
) -> TreeClassifier | None:
    """Build a supported tree model with config-driven hyperparameters."""

    normalized = model_name.strip().lower()
    if normalized not in TREE_MODEL_NAMES:
        return None

    if normalized == "random_forest":
        return RandomForestTreeClassifier(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=_normalize_optional_int(params.get("max_depth", 12)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            class_weight=_normalize_class_weight(
                params.get("class_weight", "balanced_subsample")
            ),
            n_jobs=_normalize_optional_int(params.get("n_jobs", -1)),
            random_state=random_state,
        )

    if normalized == "xgboost":
        return XGBoostTreeClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 6)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            gamma=float(params.get("gamma", 0.0)),
            early_stopping_rounds=int(params.get("early_stopping_rounds", 50)),
            eval_metric=str(params.get("eval_metric", "logloss")),
            n_jobs=_normalize_optional_int(params.get("n_jobs", -1)),
            random_state=random_state,
        )

    if normalized == "lightgbm":
        return LightGBMTreeClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 31)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 0.0)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            class_weight=_normalize_class_weight(params.get("class_weight", None)),
            early_stopping_rounds=int(params.get("early_stopping_rounds", 50)),
            n_jobs=_normalize_optional_int(params.get("n_jobs", -1)),
            random_state=random_state,
        )

    raise ValueError(f"Unsupported tree model '{normalized}'.")
