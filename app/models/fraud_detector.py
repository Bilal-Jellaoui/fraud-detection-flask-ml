import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


MODEL_DIR = Path(__file__).parent


class FraudDetector:
    """
    Wrapper autour du modèle ML sérialisé.
    Chargé une seule fois au démarrage de l'application Flask.
    """

    def __init__(self):
        self.model        = None
        self.scaler       = None
        self.meta         = {}
        self.feature_names = []
        self.is_loaded    = False

    def load(self):
        """Charge le modèle et le scaler depuis le disque."""
        model_path  = MODEL_DIR / "saved_model.pkl"
        scaler_path = MODEL_DIR / "scaler.pkl"
        meta_path   = MODEL_DIR / "model_meta.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {model_path}\n"
                "Lancez d'abord : python ml/train.py"
            )

        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            self.feature_names = self.meta.get("feature_names", [])

        self.is_loaded = True
        print(f"[FraudDetector] Modèle chargé : {self.meta.get('model_name', 'inconnu')}")
        print(f"[FraudDetector] Entraîné le   : {self.meta.get('trained_at', 'N/A')}")
        return self

    def _preprocess(self, input_data: dict) -> np.ndarray:
        if not self.feature_names:
            raise RuntimeError("feature_names vide — modèle mal chargé")
        """
        Prépare les données d'entrée pour la prédiction.
        input_data : dict avec les features de la transaction.
        """
        # Construction du vecteur de features dans le bon ordre
        feature_vector = []
        for feat in self.feature_names:
            val = input_data.get(feat, 0.0)
            feature_vector.append(float(val))

        X = np.array(feature_vector).reshape(1, -1)

        # Normalisation Amount et Time (même scaler que l'entraînement)
        if self.scaler is not None:
            cols_to_scale = ["Amount", "Time"]
            scale_indices = [self.feature_names.index(c)
                             for c in cols_to_scale
                             if c in self.feature_names]
            if scale_indices:
                X_temp = pd.DataFrame(X, columns=self.feature_names)
                X_temp[cols_to_scale] = self.scaler.transform(
                    X_temp[cols_to_scale]
                )
                X = X_temp.values

        return X

    def predict(self, input_data: dict, threshold: float = None) -> dict:
        """
        Prédit si une transaction est frauduleuse.

        Retourne :
        {
            "is_fraud": bool,
            "fraud_probability": float,
            "risk_level": str,        # "LOW" / "MEDIUM" / "HIGH"
            "confidence": float,
            "model_name": str,
        }
        """
        if not self.is_loaded:
            raise RuntimeError("Modèle non chargé. Appeler .load() d'abord.")

        threshold = threshold or self.meta.get("threshold", 0.5)
        X = self._preprocess(input_data)

        # Probabilité de fraude
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(X)[0, 1])
        else:
            raw = self.model.predict(X)[0]
            prob = float(raw == -1)  # Isolation Forest

        is_fraud = prob >= threshold

        # Niveau de risque
        if prob < 0.3:
            risk_level = "LOW"
        elif prob < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "is_fraud":          is_fraud,
            "fraud_probability": round(prob, 6),
            "risk_level":        risk_level,
            "confidence":        round(max(prob, 1 - prob), 4),
            "model_name":        self.meta.get("model_name", "Unknown"),
            "threshold":         threshold,
        }

    def get_stats(self) -> dict:
        """Retourne les statistiques du modèle (pour /stats)."""
        return {
            "model_name":   self.meta.get("model_name", "N/A"),
            "trained_at":   self.meta.get("trained_at", "N/A"),
            "metrics":      self.meta.get("metrics", {}),
            "is_loaded":    self.is_loaded,
            "n_features":   len(self.feature_names),
            "feature_names": self.feature_names,
        }


# Instance globale partagée dans Flask 
detector = FraudDetector()