"""
Exploratory Data Analysis (EDA) Module
=======================================
Functions to visualize and analyze the credit card fraud dataset.
All plots are saved to the 'assets/' directory for use in reports and dashboards.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import setup_logging, get_project_root

# Initialize logger
logger = setup_logging()

# ──────────────────────────────────────────────
#  Plot Configuration
# ──────────────────────────────────────────────

# Professional dark theme for all plots
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "legitimate": "#00D4AA",  # Teal green
    "fraud": "#FF4C6A",       # Coral red
    "primary": "#6C5CE7",     # Purple
    "secondary": "#A29BFE",   # Light purple
    "background": "#1A1A2E",  # Dark navy
    "text": "#EAEAEA",        # Light gray
}

# Global plot settings
plt.rcParams.update({
    "figure.facecolor": COLORS["background"],
    "axes.facecolor": "#16213E",
    "axes.edgecolor": "#2D3561",
    "axes.labelcolor": COLORS["text"],
    "text.color": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "figure.dpi": 150,
})


def _get_assets_path():
    """Get the absolute path to the assets directory."""
    return os.path.join(get_project_root(), "assets")


# ──────────────────────────────────────────────
#  1. Class Distribution
# ──────────────────────────────────────────────

def plot_class_distribution(df, save_path=None):
    """
    Visualize the distribution of fraudulent vs legitimate transactions.
    
    Creates a bar chart showing the severe class imbalance in the dataset.
    
    Args:
        df (pd.DataFrame): Dataset with 'Class' column
        save_path (str, optional): Custom save path. Defaults to assets/
    """
    if save_path is None:
        save_path = os.path.join(_get_assets_path(), "class_distribution.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    class_counts = df["Class"].value_counts()
    colors = [COLORS["legitimate"], COLORS["fraud"]]
    labels = ["Legitimate", "Fraudulent"]
    
    bars = axes[0].bar(labels, class_counts.values, color=colors, 
                       edgecolor="white", linewidth=0.5, width=0.6)
    axes[0].set_title("Transaction Class Distribution", fontweight="bold", pad=15)
    axes[0].set_ylabel("Number of Transactions")
    
    # Add count labels on bars
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                     f"{count:,}", ha="center", va="bottom", fontweight="bold",
                     fontsize=13, color=COLORS["text"])
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=labels, colors=colors,
                autopct="%1.3f%%", startangle=90, textprops={"fontsize": 12},
                wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                explode=(0, 0.1))
    axes[1].set_title("Fraud vs Legitimate Proportion", fontweight="bold", pad=15)
    
    plt.tight_layout(pad=3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved: class_distribution.png")


# ──────────────────────────────────────────────
#  2. Transaction Amount Distribution
# ──────────────────────────────────────────────

def plot_amount_distribution(df, save_path=None):
    """
    Plot the distribution of transaction amounts, split by class.
    
    Shows histograms with KDE overlay for both fraud and legitimate transactions.
    
    Args:
        df (pd.DataFrame): Dataset with 'Amount' and 'Class' columns
        save_path (str, optional): Custom save path
    """
    if save_path is None:
        save_path = os.path.join(_get_assets_path(), "amount_distribution.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Overall distribution
    axes[0].hist(df["Amount"], bins=100, color=COLORS["primary"], 
                 alpha=0.8, edgecolor="white", linewidth=0.3)
    axes[0].set_title("Overall Transaction Amount Distribution", fontweight="bold")
    axes[0].set_xlabel("Transaction Amount ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, df["Amount"].quantile(0.99))  # Remove extreme outliers
    
    # By class comparison
    legit = df[df["Class"] == 0]["Amount"]
    fraud = df[df["Class"] == 1]["Amount"]
    
    axes[1].hist(legit, bins=80, alpha=0.7, color=COLORS["legitimate"],
                 label=f"Legitimate (μ=${legit.mean():.2f})", 
                 edgecolor="white", linewidth=0.3, density=True)
    axes[1].hist(fraud, bins=80, alpha=0.7, color=COLORS["fraud"],
                 label=f"Fraud (μ=${fraud.mean():.2f})", 
                 edgecolor="white", linewidth=0.3, density=True)
    axes[1].set_title("Amount Distribution by Class", fontweight="bold")
    axes[1].set_xlabel("Transaction Amount ($)")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=11, framealpha=0.8)
    axes[1].set_xlim(0, 500)  # Focus on lower amounts for clarity
    
    plt.tight_layout(pad=3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved: amount_distribution.png")


# ──────────────────────────────────────────────
#  3. Time Distribution
# ──────────────────────────────────────────────

def plot_time_distribution(df, save_path=None):
    """
    Plot transaction frequency over time (seconds from first transaction).
    
    Shows how transactions are distributed over the ~48-hour period.
    
    Args:
        df (pd.DataFrame): Dataset with 'Time' and 'Class' columns
        save_path (str, optional): Custom save path
    """
    if save_path is None:
        save_path = os.path.join(_get_assets_path(), "time_distribution.png")
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Convert time to hours for readability
    df_plot = df.copy()
    df_plot["Time_Hours"] = df_plot["Time"] / 3600
    
    ax.hist(df_plot[df_plot["Class"] == 0]["Time_Hours"], bins=100,
            alpha=0.7, color=COLORS["legitimate"], label="Legitimate",
            edgecolor="white", linewidth=0.3)
    ax.hist(df_plot[df_plot["Class"] == 1]["Time_Hours"], bins=100,
            alpha=0.8, color=COLORS["fraud"], label="Fraudulent",
            edgecolor="white", linewidth=0.3)
    
    ax.set_title("Transaction Frequency Over Time", fontweight="bold", pad=15)
    ax.set_xlabel("Time (Hours from First Transaction)")
    ax.set_ylabel("Number of Transactions")
    ax.legend(fontsize=11, framealpha=0.8)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved: time_distribution.png")


# ──────────────────────────────────────────────
#  4. Correlation Heatmap
# ──────────────────────────────────────────────

def plot_correlation_heatmap(df, save_path=None):
    """
    Create a correlation heatmap of all features.
    
    Highlights which features are most correlated with the 'Class' variable.
    
    Args:
        df (pd.DataFrame): The full dataset
        save_path (str, optional): Custom save path
    """
    if save_path is None:
        save_path = os.path.join(_get_assets_path(), "correlation_heatmap.png")
    
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Compute correlation matrix
    corr = df.corr()
    
    # Use a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Custom colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=0.3, linecolor="#2D3561",
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                ax=ax, fmt=".1f",
                vmin=-1, vmax=1)
    
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", 
                 fontsize=18, pad=20)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved: correlation_heatmap.png")


# ──────────────────────────────────────────────
#  5. Top Correlated Features
# ──────────────────────────────────────────────

def plot_top_features(df, save_path=None):
    """
    Box plots of the top features most correlated with fraud.
    
    Identifies and visualizes the PCA components with strongest
    correlation to the target Class variable.
    
    Args:
        df (pd.DataFrame): The full dataset
        save_path (str, optional): Custom save path
    """
    if save_path is None:
        save_path = os.path.join(_get_assets_path(), "top_features.png")
    
    # Find top 8 features most correlated with Class
    correlations = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    top_features = correlations.head(8).index.tolist()
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        data_legit = df[df["Class"] == 0][feature]
        data_fraud = df[df["Class"] == 1][feature]
        
        bp = axes[idx].boxplot(
            [data_legit, data_fraud],
            labels=["Legitimate", "Fraud"],
            patch_artist=True,
            boxprops=dict(linewidth=1.5),
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=COLORS["text"]),
            capprops=dict(color=COLORS["text"]),
            flierprops=dict(marker="o", markersize=3, alpha=0.3),
        )
        
        bp["boxes"][0].set_facecolor(COLORS["legitimate"])
        bp["boxes"][1].set_facecolor(COLORS["fraud"])
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_alpha(0.7)
        
        corr_val = df.corr()["Class"][feature]
        axes[idx].set_title(f"{feature}\n(corr: {corr_val:.3f})", fontweight="bold")
    
    fig.suptitle("Top 8 Features Correlated with Fraud", fontweight="bold",
                 fontsize=18, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved: top_features.png")


# ──────────────────────────────────────────────
#  6. Generate Insights
# ──────────────────────────────────────────────

def generate_insights(df):
    """
    Generate and display key statistical insights from the dataset.
    
    Prints actionable findings about fraud patterns, transaction amounts,
    and feature characteristics.
    
    Args:
        df (pd.DataFrame): The full dataset
    
    Returns:
        list[str]: List of insight strings
    """
    insights = []
    
    total = len(df)
    fraud_count = df["Class"].sum()
    legit_count = total - fraud_count
    
    # Insight 1: Class imbalance
    insights.append(
        f"📊 SEVERE CLASS IMBALANCE: Only {fraud_count:,} ({fraud_count/total*100:.3f}%) "
        f"of {total:,} transactions are fraudulent."
    )
    
    # Insight 2: Amount comparison
    fraud_amt = df[df["Class"] == 1]["Amount"]
    legit_amt = df[df["Class"] == 0]["Amount"]
    insights.append(
        f"💰 FRAUD AMOUNT: Mean fraud amount (${fraud_amt.mean():.2f}) vs "
        f"legitimate (${legit_amt.mean():.2f}). "
        f"Max fraud: ${fraud_amt.max():.2f}."
    )
    
    # Insight 3: Small fraud transactions
    small_fraud = (fraud_amt < 100).sum()
    insights.append(
        f"🔍 SMALL TRANSACTIONS: {small_fraud} ({small_fraud/fraud_count*100:.1f}%) "
        f"of fraud transactions are under $100, making them harder to detect."
    )
    
    # Insight 4: Top correlated features
    correlations = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    top3 = correlations.head(3)
    top3_str = ", ".join([f"{name} ({val:.3f})" for name, val in top3.items()])
    insights.append(
        f"🔗 TOP CORRELATED FEATURES: {top3_str}"
    )
    
    # Insight 5: Time patterns
    df_hours = df.copy()
    df_hours["Hour"] = (df_hours["Time"] / 3600).astype(int) % 24
    fraud_by_hour = df_hours[df_hours["Class"] == 1]["Hour"].value_counts()
    peak_hour = fraud_by_hour.index[0]
    insights.append(
        f"⏰ TIME PATTERN: Fraud peaks around hour {peak_hour} "
        f"from the start of recording."
    )
    
    # Insight 6: No missing values typically
    missing = df.isnull().sum().sum()
    insights.append(
        f"✅ DATA QUALITY: {'No' if missing == 0 else missing} missing values found. "
        f"Dataset is {'clean' if missing == 0 else 'requires cleaning'}."
    )
    
    # Print all insights
    logger.info("\n" + "=" * 60)
    logger.info("KEY INSIGHTS FROM EDA")
    logger.info("=" * 60)
    for i, insight in enumerate(insights, 1):
        logger.info(f"  {i}. {insight}")
    logger.info("=" * 60)
    
    return insights


# ──────────────────────────────────────────────
#  Full EDA Pipeline
# ──────────────────────────────────────────────

def run_full_eda(df):
    """
    Execute all EDA visualizations and generate insights.
    
    Creates all plots and saves them to the assets/ directory.
    
    Args:
        df (pd.DataFrame): The dataset (preferably before scaling)
    
    Returns:
        list[str]: Generated insights
    """
    logger.info("=" * 60)
    logger.info("STARTING EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)
    
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_time_distribution(df)
    plot_correlation_heatmap(df)
    plot_top_features(df)
    insights = generate_insights(df)
    
    logger.info("EDA completed — All plots saved to assets/")
    return insights
