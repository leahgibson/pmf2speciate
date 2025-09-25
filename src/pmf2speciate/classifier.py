"""
Factor classification using EPA SPECIATE
"""

import os
import pandas as pd
import numpy as np
import pickle
from typing import Dict
from pathlib import Path


class SourceClassifier:
    def __init__(self):
        """
        Initialize the source classifier with trained models.

        Parameters:
        models_dir: Path to directory contianing the trained models.
        """

        self.models_dir = Path(__file__).parent / "models"
        self.generation_model = None
        self.source_models = {}
        self.feature_names = {}
        self.mechanisms = ["Combustion", "Microbial", "Volatilization"]

        self._load_models()

    def _load_models(self):
        """Load trained models and feature names"""

        # Load generation mechanism model
        gen_model_path = os.path.join(
            self.models_dir, "generation_mechanism_random_forest.pkl"
        )
        gen_features_path = os.path.join(
            self.models_dir, "generation_mechanism_feature_names.pkl"
        )

        if os.path.exists(gen_model_path) and os.path.exists(gen_features_path):
            with open(gen_model_path, "rb") as f:
                self.generation_model = pickle.load(f)
            with open(gen_features_path, "rb") as f:
                self.feature_names["generation_mechanism"] = pickle.load(f)
        else:
            raise FileNotFoundError("Generation mechanism model files not found.")

        # Load individual source models for each mechanism
        for mechanism in self.mechanisms:
            model_path = os.path.join(self.models_dir, f"{mechanism}_random_forest.pkl")
            features_path = os.path.join(
                self.models_dir, f"{mechanism}_feature_names.pkl"
            )

            if os.path.exists(model_path) and os.path.exists(features_path):
                with open(model_path, "rb") as f:
                    self.source_models[mechanism] = pickle.load(f)
                with open(features_path, "rb") as f:
                    self.feature_names[mechanism] = pickle.load(f)
            else:
                raise FileNotFoundError(f"{mechanism} model files not found")

    def _prepare_features(
        self, profile: Dict[str, float], feature_names: pd.Index
    ) -> np.ndarray:
        """
        Converts CAS profile dict to feature array, matching model expectations.

        Parameters:
        profile: Dict
            CAS numbers as keys and percent weights as values
        feature_names: pd.Index
            Expected featyre names from the model

        Returs:
            Feature array ready for prediction
        """

        features = dict.fromkeys(feature_names, 0.0)
        for i, feature_name in enumerate(feature_names):
            features[feature_name] = profile.get(feature_name, 0.0)

        return pd.DataFrame(features, index=[0])

    def identify_source(self, profile: Dict[str, float]) -> Dict:
        """
        Identify the source of a factor profile using 2-tier hierarchical classification.

        Parameters:
        profile: Dict
            CAS numbers as keys and percent weights as values

            Returns: Dict containing
            - generation_mechanism: Predicted generation mechanism
            - generation_confidence: Confidence for generation mechanism prediction
            - specific_source: Predicted specific source (if available)
            - source_confidence: Confidence for specific source prediction (if available)
            - overall_confidence: Combined confidence score
        """

        if self.generation_model is None:
            raise RuntimeError("Models not loaded.")

        # Tier 1: predict generation mechanism
        gen_features = self._prepare_features(
            profile, self.feature_names["generation_mechanism"]
        )
        gen_prediction = self.generation_model.predict(gen_features)[0]
        gen_probabilities = self.generation_model.predict_proba(gen_features)[0]
        gen_confidence = np.max(gen_probabilities)

        result = {
            "generation_mechanism": gen_prediction,
            "generation_confidence": float(gen_confidence),
            "specific_source": None,
            "source_confidence": None,
            "overall_confidence": float(gen_confidence),
        }

        # Tier 2: predict source using the predicted generation mechanism
        if gen_prediction in self.source_models:
            source_model = self.source_models[gen_prediction]
            source_features = self._prepare_features(
                profile, self.feature_names[gen_prediction]
            )

            source_prediction = source_model.predict(source_features)[0]
            source_probabilities = source_model.predict_proba(source_features)[0]
            source_confidence = np.max(source_probabilities)

            result.update(
                {
                    "specific_source": source_prediction,
                    "source_confidence": float(source_confidence),
                    "overall_confidence": float(gen_confidence * source_confidence),
                }
            )

        return result

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        info = {
            "generation_mechanism": {
                "loaded": self.generation_model is not None,
                "classes": self.generation_model.classes_.tolist()
                if self.generation_model
                else None,
            }
        }

        for mechanism in self.mechanisms:
            info[mechanism] = {
                "loaded": mechanism in self.source_models,
                "classes": self.source_models[mechanism].classes_.tolist()
                if mechanism in self.source_models
                else None,
            }

        return info
