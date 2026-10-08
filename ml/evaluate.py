import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, precision_recall_curve,
    average_precision_score
)

OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 1. Évaluation d'un modèle unique
def evaluate_model(model, X_test, y_test, model_name: str = "Model",
                   threshold: float = 0.5) -> dict:
    """
    Calcule toutes les métriques demandées dans mon projet.
    Retourne un dict : Accuracy, Precision, Recall, F1-Score, ROC-AUC.
    """
    # Prédictions
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_prob = y_pred.astype(float)

    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred),
        "F1-Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
    }

    print(f"\n── Métriques : {model_name} ──")
    for k, v in metrics.items():
        print(f"   {k:<12}: {v:.4f}")

    print(f"\n── Rapport de classification ──")
    print(classification_report(y_test, y_pred,
          target_names=["Légitime (0)", "Fraude (1)"]))

    return metrics


# 2. Tableau  style heatmap verte )

def compare_models(results: dict, save: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    df.index.name = "Modèle"
    df = df[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    df = df.sort_values("ROC-AUC", ascending=False)

    fig, ax = plt.subplots(figsize=(10, len(df) * 0.7 + 1.5))
    ax.axis("off")

    # Seaborn heatmap
    import seaborn as sns
    ax2 = fig.add_axes([0.15, 0.05, 0.82, 0.88])
    sns.heatmap(
        df.astype(float),
        annot=True, fmt=".4f",
        cmap="Blues",
        linewidths=0.5,
        ax=ax2,
        vmin=0, vmax=1,
        annot_kws={"size": 11}
    )
    ax2.set_title("modèles", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("")
    ax2.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "model.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[OK] Tableau → {path}")
    plt.show()
    return df



# 3. Courbes ROC + Precision-Recall 
def plot_roc_curves(models: dict, X_test, y_test,
                    iso_predict_fn=None, save: bool = True):
    """
    Courbes ROC-AUC et Precision-Recall pour tous les modèles.
    Style.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    colors = {
        "Logistic Regression": "#1f77b4",
        "Random Forest":       "#d62728",
        "XGBoost":             "#ff7f0e",
        "Isolation Forest":    "#9467bd",
    }

    # ── Graphe 1 : Courbes ROC ──
    ax_roc = axes[0]
    ax_roc.set_facecolor("white")
    ax_roc.grid(True, alpha=0.3, color="#cccccc")

    for name, model in models.items():
        if name == "Isolation Forest" and iso_predict_fn is not None:
            y_score = iso_predict_fn(model, X_test).astype(float)
        elif hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_val = roc_auc_score(y_test, y_score)
        color = colors.get(name, "#333333")
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name} (AUC = {auc_val:.4f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_roc.set_xlabel("Taux de Faux Positifs", fontsize=11)
    ax_roc.set_ylabel("Taux de Vrais Positifs (Recall)", fontsize=11)
    ax_roc.set_title("Comparaison des courbes ROC-AUC", fontsize=13, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([-0.01, 1.05])

    # ── Graphe 2 : Precision-Recall ──
    ax_pr = axes[1]
    ax_pr.set_facecolor("white")
    ax_pr.grid(True, alpha=0.3, color="#cccccc")

    for name, model in models.items():
        if name == "Isolation Forest" and iso_predict_fn is not None:
            y_score = iso_predict_fn(model, X_test).astype(float)
        elif hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            continue

        prec, rec, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        color = colors.get(name, "#333333")
        ax_pr.plot(rec, prec, color=color, lw=2,
                   label=f"{name} (AP = {ap:.3f})")

    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("Comparaison des courbes Precision-Recall", fontsize=13, fontweight="bold")
    ax_pr.legend(loc="upper right", fontsize=9)
    ax_pr.set_xlim([-0.01, 1.01])
    ax_pr.set_ylim([-0.01, 1.05])

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "roc_pr_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[OK] Courbes ROC/PR sauvegardées → {path}")
    plt.show()



# 4. Matrices de confusion — style bleu
def plot_confusion_matrices(models: dict, X_test, y_test,
                              iso_predict_fn=None, save: bool = True):
    """
    Matrices de confusion style bleu identique à l'image 3.
    """
    n = len(models)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten()
    fig.patch.set_facecolor("white")

    # Palette bleue 
    blue_cmap = mcolors.LinearSegmentedColormap.from_list(
        "blues_fraud",
        ["#eaf4fb", "#a8d4ea", "#2c5f8a", "#0d2d52"],
        N=256
    )

    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]

        if name == "Isolation Forest" and iso_predict_fn is not None:
            y_pred = iso_predict_fn(model, X_test)
        elif hasattr(model, "predict_proba"):
            y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        # Plot heatmap 
        sns.heatmap(
            cm, ax=ax,
            cmap=blue_cmap,
            annot=True, fmt="d",
            annot_kws={"size": 14, "weight": "bold", "color": "#2c5f8a"},
            linewidths=0.5,
            linecolor="#dddddd",
            cbar=False,
            square=True
        )

        ax.set_title(name, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)
        ax.set_xticklabels(["0", "1"], fontsize=10)
        ax.set_yticklabels(["0", "1"], fontsize=10, rotation=0)

        # Stats sous la matrice
        tn, fp, fn, tp = cm.ravel()
        ax.text(0.5, -0.15,
                f"TN={tn:,}  FP={fp}  FN={fn}  TP={tp}",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=9, color="#555555")

    # Masquer les axes vides
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Matrices de confusion — Tous les modèles",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "confusion_matrices.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[OK] Matrices sauvegardées → {path}")
    plt.show()



# 5. Feature Importance + SHAP
def plot_feature_importance(model, feature_names: list,
                              model_name: str = "Random Forest",
                              top_n: int = 20, save: bool = True):
    """Importance des features pour RF/XGBoost."""
    if not hasattr(model, "feature_importances_"):
        print(f"[WARN] {model_name} ne supporte pas feature_importances_")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("white")
    ax.grid(axis="x", alpha=0.3)

    colors_bar = plt.cm.Greens(
        np.linspace(0.4, 0.9, top_n)
    )[::-1]

    bars = ax.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color=colors_bar,
        edgecolor="none"
    )

    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Top {top_n} features — {model_name}",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / f"feature_importance_{model_name.replace(' ','_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[OK] Feature importance → {path}")
    plt.show()


def plot_shap_values(model, X_test, feature_names: list,
                      model_name: str = "Random Forest",
                      max_display: int = 20, save: bool = True):
    import shap

    # Échantillon pour la rapidité
    sample_size = min(500, len(X_test))
    X_sample = X_test.iloc[:sample_size] if hasattr(X_test, "iloc") else X_test[:sample_size]

    try:
        if hasattr(model, "predict_proba"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # Logistic Regression → LinearExplainer
            explainer  = shap.LinearExplainer(model, X_sample)
            shap_vals  = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"[WARN] SHAP non disponible : {e}")
        return

    # Pour RF binaire, prendre classe 1
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap.summary_plot(
        shap_vals, X_sample,
        feature_names=feature_names,
        max_display=20, show=False
    )
    plt.title(f"SHAP Values — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / f"shap_{model_name.replace(' ','_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()