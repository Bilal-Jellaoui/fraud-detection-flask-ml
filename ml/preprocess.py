import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import joblib


DATA_PATH = Path(__file__).parent.parent / "data" / "creditcard.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Charge le dataset et affiche un résumé basique."""
    df = pd.read_csv(path)
    print(f"[INFO] Dataset chargé : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    print(f"[INFO] Fraudes : {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Gestion des valeurs manquantes (remplacement par médiane)."""
    missing = df.isnull().sum()
    if missing.any():
        print(f"[WARN] Valeurs manquantes détectées :\n{missing[missing > 0]}")
        df = df.fillna(df.median(numeric_only=True))
    else:
        print("[INFO] Aucune valeur manquante")
    return df


def get_features_target(df: pd.DataFrame):
    """Sépare les features (X) de la cible (y)."""
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].copy()
    y = df["Class"].copy()
    return X, y


def scale_features(X_train, X_test):
    """
    Normalise Amount et Time avec StandardScaler.
    Les features V1-V28 (PCA) sont déjà normalisées.
    Retourne X_train_scaled, X_test_scaled, scaler.
    """
    scaler = StandardScaler()
    cols_to_scale = ["Amount", "Time"]

    X_train = X_train.copy()
    X_test  = X_test.copy()

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

    # Sauvegarde du scaler pour l'API Flask
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print("[INFO] Scaler sauvegardé → app/models/scaler.pkl")

    return X_train, X_test, scaler


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split stratifié 80/20."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"[INFO] Train : {X_train.shape[0]:,} | Test : {X_test.shape[0]:,}")
    print(f"[INFO] Fraudes train : {y_train.sum()} | Fraudes test : {y_test.sum()}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42,
                sampling_strategy: float = 0.1):
    """
    SMOTE appliqué uniquement sur les données d'entraînement.
    sampling_strategy=0.1 : ratio fraudes/légitimes = 10% après oversampling.
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=5
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    # Reconvertir en DataFrame pour garder les noms de colonnes
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name="Class")
    return X_res, y_res


def apply_nearmiss(X_train, y_train, version: int = 1):
    """Under-sampling NearMiss (alternative à SMOTE)."""
    nm = NearMiss(version=version)
    X_res, y_res = nm.fit_resample(X_train, y_train)
    print(f"[INFO] Après NearMiss → {X_res.shape[0]:,} échantillons")
    return X_res, y_res


def full_pipeline(sampling: str = "smote") -> dict:
    """
    Pipeline complet : load → clean → split → scale → resample.
    Retourne un dict avec toutes les données prêtes pour l'entraînement.
    """
    df = load_data()
    df = handle_missing(df)
    X, y = get_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test, scaler = scale_features(X_train, X_test)

    if sampling == "smote":
        X_train_res, y_train_res = apply_smote(X_train, y_train)
    elif sampling == "nearmiss":
        X_train_res, y_train_res = apply_nearmiss(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    return {
        "X_train": X_train_res,
        "X_test":  X_test,
        "y_train": y_train_res,
        "y_test":  y_test,
        "scaler":  scaler,
        "feature_names": list(X.columns),
    }


if __name__ == "__main__":
    data = full_pipeline(sampling="smote")
    print("\n[OK] Pipeline terminé.")
    print(f"     X_train : {data['X_train'].shape}")
    print(f"     X_test  : {data['X_test'].shape}")