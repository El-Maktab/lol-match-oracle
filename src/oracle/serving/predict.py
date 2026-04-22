import glob
import json
import os
import pickle
from typing import Dict, List

import pandas as pd

from .schemas import PredictRequest, PredictResponse


class PredictionService:
    def __init__(self, model_dir: str = None, preprocessor_path: str = None):
        """
        Initialize the PredictionService. 
        Loads the preprocessor and the champion model.
        """
        # Determine paths if not provided
        if not preprocessor_path:
            preprocessor_path = os.environ.get(
                "PREPROCESSOR_PATH", 
                "data/processed/feature_preprocessor.pkl"
            )
        
        if not model_dir:
            model_dir = os.environ.get("MODEL_DIR", "")
            if not model_dir:
                # Find the champion model dynamically
                # According to docs, the champion is LightGBM optuna-best-model
                model_metadata_paths = glob.glob(
                    "models/02-advanced-models/lightgbm/*/model_metadata.json"
                )
                for path in model_metadata_paths:
                    with open(path, "r") as f:
                        meta = json.load(f)
                    if meta.get("run_name") == "optuna-best-model":
                        model_dir = os.path.dirname(path)
                        break

        if not model_dir or not os.path.exists(model_dir):
            raise RuntimeError(f"Could not find model directory: {model_dir}")

        if not os.path.exists(preprocessor_path):
            raise RuntimeError(f"Could not find preprocessor: {preprocessor_path}")

        # Load preprocessor
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

        # Load model
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load metadata
        meta_path = os.path.join(model_dir, "model_metadata.json")
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
            
        self.feature_columns = self.metadata.get("feature_columns", [])
        self.model_version = self.metadata.get("run_name", "unknown")

    def _preprocess(self, features_list: List[Dict[str, float]]) -> pd.DataFrame:
        """
        Convert dicts to DataFrame and apply the trained ColumnTransformer.
        
        LFD Anchor: Production inference must preserve training-time hypothesis 
        assumptions. We guarantee this by applying the exact same scaling statistics
        (from the fitted ColumnTransformer) and aligning features to the subset
        the model was validated against.
        """
        # Create DataFrame
        df = pd.DataFrame(features_list)
        
        # We need all features that the preprocessor expects.
        # We can get them from the transformers inside the preprocessor.
        preprocessor_cols = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name != "remainder":
                preprocessor_cols.extend(columns)
                
        # Ensure all required columns for the preprocessor exist
        for col in preprocessor_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        # The preprocessor expects the exact columns defined during training
        x_scaled = self.preprocessor.transform(df[preprocessor_cols])
        
        # The preprocessor returns a numpy array. We convert it back to a DataFrame 
        # using the output feature names so we can select the columns the model expects.
        out_col_names = [
            name.split("__", 1)[-1] 
            for name in self.preprocessor.get_feature_names_out()
        ]
        df_scaled = pd.DataFrame(x_scaled, columns=out_col_names, index=df.index)
        
        # Finally, slice the DataFrame to match the features the model was trained on
        for col in self.feature_columns:
            if col not in df_scaled.columns:
                df_scaled[col] = 0.0
                
        return df_scaled[self.feature_columns]

    def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Single prediction.
        """
        features_dict = request.features.model_dump()
        x_scaled = self._preprocess([features_dict])
        
        # model.predict_proba returns shape (n_samples, n_classes)
        probs = self.model.predict_proba(x_scaled)[0]
        
        # Class 1 is win, Class 0 is loss
        win_prob = float(probs[1])
        prediction = 1 if win_prob >= 0.5 else 0
        confidence = win_prob if prediction == 1 else float(probs[0])

        return PredictResponse(
            prediction=prediction,
            confidence=confidence,
            win_probability=win_prob,
            model_version=self.model_version
        )

    def predict_batch(self, requests: List[PredictRequest]) -> List[PredictResponse]:
        """
        Batch prediction.
        """
        features_list = [req.features.model_dump() for req in requests]
        x_scaled = self._preprocess(features_list)
        
        probs = self.model.predict_proba(x_scaled)
        
        responses = []
        for p in probs:
            win_prob = float(p[1])
            prediction = 1 if win_prob >= 0.5 else 0
            confidence = win_prob if prediction == 1 else float(p[0])
            
            responses.append(PredictResponse(
                prediction=prediction,
                confidence=confidence,
                win_probability=win_prob,
                model_version=self.model_version
            ))
            
        return responses

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return the feature importance from the underlying model.
        """
        # The TreeClassifier wrapper exposes feature_importances_
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            # The preprocessor might have dropped or rearranged columns.
            # Assuming the order matches the preprocessor's output names
            feature_names = [
                name.split("__", 1)[-1] 
                for name in self.preprocessor.get_feature_names_out()
            ]
            
            if len(importances) == len(feature_names):
                return {name: float(imp) for name, imp in zip(feature_names, importances)}
                
        return {}
