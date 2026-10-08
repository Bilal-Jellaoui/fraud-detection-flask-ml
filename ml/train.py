"""
Entraînement, comparaison et sauvegarde du meilleur modèle
Modèles : Logistic Regression, Random Forest, XGBoost, Isolation Forest
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from datetime import datetime

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, IsolationForest
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost                 import XGBClassifier

from preprocess import full_pipeline
from evaluate   import evaluate_model, compare_models, plot_roc_curves, plot_confusion_matrices

MODEL_DIR = Path(__file__).parent.parent / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Définition des modèles et grilles d'hyperparamètres
MODELS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs"
        ),
        "params": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
        },
        "search": True
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        "params": {
            "n_estimators":      [50, 100, 150],
            "max_depth":         [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "max_features":      ["sqrt", "log2"],
        },
        "search": True
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        ),
        "params": {
            "n_estimators":  [50, 100, 150],
            "max_depth":     [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample":     [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "scale_pos_weight": [1, 10, 50, 100],
        },
        "search": True
    },
}


def train_with_search(name: str, config: dict, X_train, y_train,
                       n_iter: int = 7, cv: int = 5) -> object:
    """
    RandomizedSearchCV avec validation croisée stratifiée.
    Optimise sur le score F1 (adapté aux données déséquilibrées).
    """
    print(f"\n{'='*55}")
    print(f"  Entraînement : {name}")
    print(f"{'='*55}")

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    if config["search"] and config["params"]:
        search = RandomizedSearchCV(
            estimator  = config["estimator"],
            param_distributions = config["params"],
            n_iter     = n_iter,
            cv         = cv_strategy,
            scoring    = "f1",
            n_jobs     = -1,
            random_state = 42,
            verbose    = 1,
            refit      = True,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"  Meilleurs paramètres : {search.best_params_}")
        print(f"  Meilleur F1 CV       : {search.best_score_:.4f}")
    else:
        best_model = config["estimator"]
        best_model.fit(X_train, y_train)

    return best_model


def train_isolation_forest(X_train, y_train):
    """
    Isolation Forest — détection d'anomalies non supervisée.
    Entraîné uniquement sur les données légitimes (classe 0).
    """
    print(f"\n{'='*55}")
    print(f"  Entraînement : Isolation Forest (anomaly detection)")
    print(f"{'='*55}")

    X_legit = X_train[y_train == 0]
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.001,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_legit)
    print(f"  Entraîné sur {len(X_legit):,} transactions légitimes")
    return iso


def predict_isolation_forest(iso_model, X):
    """Convertit les prédictions IF (-1/1) en (1/0) pour compatibilité sklearn."""
    raw = iso_model.predict(X)
    return np.where(raw == -1, 1, 0)


def save_model_artifacts(best_model, model_name: str, metrics: dict,
                          feature_names: list):
    """Sauvegarde le modèle, ses métadonnées et les noms de features."""
    # Modèle principal
    joblib.dump(best_model, MODEL_DIR / "saved_model.pkl")

    # Métadonnées JSON (utilisées par l'API Flask)
    meta = {
        "model_name":    model_name,
        "trained_at":    datetime.now().isoformat(),
        "metrics":       metrics,
        "feature_names": feature_names,
        "threshold":     0.5,
    }
    with open(MODEL_DIR / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Modèle sauvegardé → app/models/saved_model.pkl")
    print(f"[OK] Métadonnées       → app/models/model_meta.json")


def run_training_pipeline():
    """Pipeline complet d'entraînement."""
    print("\n" + "="*55)
    print("  PIPELINE ML — Détection de Fraude Bancaire")
    print("="*55)

    # 1. Prétraitement
    data = full_pipeline(sampling="smote")
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    feature_names = data["feature_names"]

    results    = {}
    trained_models = {}

    # 2. Entraînement des modèles supervisés
    for name, config in MODELS.items():
        model = train_with_search(name, config, X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results[name] = metrics
        trained_models[name] = model

    # 3. Isolation Forest (non supervisé)
    iso_model = train_isolation_forest(X_train, y_train)
    y_pred_iso = predict_isolation_forest(iso_model, X_test)
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, roc_auc_score)
    iso_metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred_iso),
        "Precision": precision_score(y_test, y_pred_iso, zero_division=0),
        "Recall":    recall_score(y_test, y_pred_iso),
        "F1-Score":  f1_score(y_test, y_pred_iso),
        "ROC-AUC":   roc_auc_score(y_test, y_pred_iso),
    }
    results["Isolation Forest"] = iso_metrics
    trained_models["Isolation Forest"] = iso_model

    # 4. Tableau comparatif (style heatmap comme ton image)
    print("\n\n" + "="*55)
    print("  COMPARAISON DES MODÈLES")
    print("="*55)
    compare_models(results)

    # 5. Courbes ROC + matrices de confusion
    plot_roc_curves(trained_models, X_test, y_test,
                    iso_predict_fn=predict_isolation_forest)
    plot_confusion_matrices(trained_models, X_test, y_test,
                             iso_predict_fn=predict_isolation_forest)

    # 6. Sélection du meilleur modèle (AUC-ROC max)
    supervised_results = {k: v for k, v in results.items()
                          if k != "Isolation Forest"}
    best_name = max(supervised_results, key=lambda k: supervised_results[k]["ROC-AUC"])
    best_metrics = results[best_name]
    best_model_obj = trained_models[best_name]

    print(f"\n{'='*55}")
    print(f"  MEILLEUR MODÈLE : {best_name}")
    print(f"  AUC-ROC  : {best_metrics['ROC-AUC']:.4f}")
    print(f"  F1-Score : {best_metrics['F1-Score']:.4f}")
    print(f"  Précision: {best_metrics['Precision']:.4f}")
    print(f"  Rappel   : {best_metrics['Recall']:.4f}")
    print(f"{'='*55}")

    # 7. Sauvegarde
    save_model_artifacts(best_model_obj, best_name, best_metrics, feature_names)

    return trained_models, results, X_test, y_test, feature_names


if __name__ == "__main__":
    run_training_pipeline()