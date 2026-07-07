"""
Model Training & Evaluation Module
====================================
Handles training multiple ML models, comprehensive evaluation,
model comparison, visualization, and saving the best model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from src.utils import setup_logging, save_object, get_project_root

# Initialize logger
logger = setup_logging()

# Plot colors (consistent with EDA module)
COLORS = {
    "legitimate": "#00D4AA",
    "fraud": "#FF4C6A",
    "primary": "#6C5CE7",
    "secondary": "#A29BFE",
    "background": "#1A1A2E",
    "text": "#EAEAEA",
}

MODEL_COLORS = {
    "Logistic Regression": "#6C5CE7",
    "Decision Tree": "#00D4AA",
    "Random Forest": "#FDCB6E",
    "XGBoost": "#FF4C6A",
}


# ──────────────────────────────────────────────
#  Model Definitions
# ──────────────────────────────────────────────

def get_models():
    """
    Get a dictionary of pre-configured ML models for training.
    
    Returns:
        dict: {model_name: model_instance}
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
        ),
    }
    
    return models


# ──────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────

def train_model(model, X_train, y_train, model_name="Model"):
    """
    Train a single ML model on the training data.
    
    Args:
        model: Scikit-learn compatible model instance
        X_train: Training features
        y_train: Training labels
        model_name (str): Name for logging
    
    Returns:
        Trained model instance
    """
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    logger.info(f"  ✓ {model_name} trained successfully")
    return model


# ──────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model with comprehensive metrics.
    
    Args:
        model: Trained model instance
        X_test: Test features
        y_test: Test labels
        model_name (str): Name for logging
    
    Returns:
        dict: Evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    
    logger.info(f"\n  📊 {model_name} Results:")
    for metric, value in metrics.items():
        logger.info(f"     {metric:<12}: {value:.4f}")
    
    return metrics


# ──────────────────────────────────────────────
#  Confusion Matrix Plot
# ──────────────────────────────────────────────

def plot_confusion_matrices(trained_models, X_test, y_test, save_path=None):
    """
    Plot confusion matrices for all trained models side by side.
    
    Args:
        trained_models (dict): {name: trained_model}
        X_test: Test features
        y_test: Test labels
        save_path (str, optional): Custom save path
    """
    if save_path is None:
        save_path = os.path.join(get_project_root(), "assets", "confusion_matrices.png")
    
    n_models = len(trained_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt=",d", cmap="YlOrRd",
                    xticklabels=["Legitimate", "Fraud"],
                    yticklabels=["Legitimate", "Fraud"],
                    ax=axes[idx], linewidths=1, linecolor="#2D3561",
                    cbar_kws={"shrink": 0.8})
        
        axes[idx].set_title(f"{name}", fontweight="bold", fontsize=13)
        axes[idx].set_ylabel("Actual" if idx == 0 else "")
        axes[idx].set_xlabel("Predicted")
    
    fig.suptitle("Confusion Matrices — All Models", fontweight="bold",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=COLORS["background"],
                dpi=150)
    plt.close(fig)
    logger.info("Saved: confusion_matrices.png")


# ──────────────────────────────────────────────
#  ROC Curves
# ──────────────────────────────────────────────

def plot_roc_curves(trained_models, X_test, y_test, save_path=None):
    """
    Plot comparative ROC curves for all trained models.
    
    Args:
        trained_models (dict): {name: trained_model}
        X_test: Test features
        y_test: Test labels
        save_path (str, optional): Custom save path
    """
    if save_path is None:
        save_path = os.path.join(get_project_root(), "assets", "roc_curves.png")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        color = MODEL_COLORS.get(name, COLORS["primary"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})",
                linewidth=2.5, color=color, alpha=0.9)
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.5, label="Random (AUC = 0.5)")
    
    ax.set_title("ROC Curves — Model Comparison", fontweight="bold", fontsize=16, pad=15)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.8)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=COLORS["background"],
                dpi=150)
    plt.close(fig)
    logger.info("Saved: roc_curves.png")


# ──────────────────────────────────────────────
#  Model Comparison Bar Chart
# ──────────────────────────────────────────────

def plot_model_comparison(all_results, save_path=None):
    """
    Create a grouped bar chart comparing all models across metrics.
    
    Args:
        all_results (dict): {model_name: {metric_name: value}}
        save_path (str, optional): Custom save path
    """
    if save_path is None:
        save_path = os.path.join(get_project_root(), "assets", "model_comparison.png")
    
    df = pd.DataFrame(all_results).T
    metrics = df.columns.tolist()
    models = df.index.tolist()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(metrics))
    width = 0.18
    
    for i, model in enumerate(models):
        color = MODEL_COLORS.get(model, COLORS["primary"])
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, df.loc[model].values, width,
                      label=model, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                    f"{height:.3f}", ha="center", va="bottom",
                    fontsize=8, color=COLORS["text"], fontweight="bold")
    
    ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=16, pad=15)
    ax.set_xlabel("Evaluation Metrics", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10, framealpha=0.8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=COLORS["background"],
                dpi=150)
    plt.close(fig)
    logger.info("Saved: model_comparison.png")


# ──────────────────────────────────────────────
#  Best Model Selection
# ──────────────────────────────────────────────

def select_best_model(all_results, trained_models):
    """
    Select the best model based on F1 Score (most appropriate for
    imbalanced classification problems).
    
    Args:
        all_results (dict): {model_name: metrics_dict}
        trained_models (dict): {model_name: trained_model}
    
    Returns:
        tuple: (best_model_name, best_model, best_metrics)
    """
    best_name = max(all_results, key=lambda k: all_results[k]["F1 Score"])
    best_model = trained_models[best_name]
    best_metrics = all_results[best_name]
    
    logger.info("\n" + "=" * 60)
    logger.info(f"🏆 BEST MODEL: {best_name}")
    logger.info(f"   F1 Score: {best_metrics['F1 Score']:.4f}")
    logger.info(f"   ROC-AUC:  {best_metrics['ROC-AUC']:.4f}")
    logger.info("=" * 60)
    
    return best_name, best_model, best_metrics


# ──────────────────────────────────────────────
#  Full Training Pipeline
# ──────────────────────────────────────────────

def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """
    Train all models, evaluate them, generate visualizations, 
    and select the best one.
    
    Args:
        X_train: Training features (SMOTE-balanced)
        X_test: Test features
        y_train: Training labels (SMOTE-balanced)
        y_test: Test labels
    
    Returns:
        dict: Contains trained_models, all_results, best_model_name,
              best_model, and best_metrics
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING & EVALUATION")
    logger.info("=" * 60)
    
    models = get_models()
    trained_models = {}
    all_results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        trained_model = train_model(model, X_train, y_train, name)
        metrics = evaluate_model(trained_model, X_test, y_test, name)
        trained_models[name] = trained_model
        all_results[name] = metrics
    
    # Generate visualizations
    logger.info("\nGenerating evaluation plots...")
    plot_confusion_matrices(trained_models, X_test, y_test)
    plot_roc_curves(trained_models, X_test, y_test)
    plot_model_comparison(all_results)
    
    # Select best model
    best_name, best_model, best_metrics = select_best_model(all_results, trained_models)
    
    return {
        "trained_models": trained_models,
        "all_results": all_results,
        "best_model_name": best_name,
        "best_model": best_model,
        "best_metrics": best_metrics,
    }
