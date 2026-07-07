# 🛡️ Credit Card Fraud Detection using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006EFF?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**An end-to-end Machine Learning project for detecting fraudulent credit card transactions**

[Getting Started](#-getting-started) · [Project Structure](#-project-structure) · [How It Works](#-how-it-works) · [Dashboard](#-streamlit-dashboard)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Screenshots](#-screenshots)
- [License](#-license)

---

## 🎯 Overview

Credit card fraud is a significant financial threat affecting millions of transactions globally. This project builds a **complete Machine Learning pipeline** to automatically detect fraudulent transactions using the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset contains **284,807 transactions** made by European cardholders in September 2013, with only **492 frauds (0.172%)**, making it a highly imbalanced classification problem.

### Key Highlights

- 🧹 Complete data preprocessing with SMOTE for class balancing
- 📊 Comprehensive EDA with 5+ professional visualizations
- 🤖 4 ML models trained and compared (LR, DT, RF, XGBoost)
- 📈 Multi-metric evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
- 🌐 Interactive Streamlit dashboard with real-time predictions
- 📄 University-ready documentation and presentation points

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Data Preprocessing** | Missing value handling, duplicate removal, StandardScaler normalization |
| **SMOTE Balancing** | Synthetic oversampling applied to training data only (prevents data leakage) |
| **Exploratory Data Analysis** | Class distribution, amount histograms, time series, correlation heatmaps |
| **Multi-Model Training** | Logistic Regression, Decision Tree, Random Forest, XGBoost |
| **Comprehensive Evaluation** | Confusion matrices, ROC curves, model comparison charts |
| **Best Model Selection** | Automatic selection based on F1 Score |
| **Streamlit Dashboard** | Dark-themed UI with fraud prediction, gauge charts, and analytics |
| **Modular Architecture** | Clean, reusable, well-documented Python modules |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **Data** | Pandas, NumPy |
| **ML** | Scikit-learn, XGBoost |
| **Balancing** | Imbalanced-learn (SMOTE) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Serialization** | Pickle |

---

## 📁 Project Structure

```
CreditCardFraudDetection/
│
├── dataset/                      # Place creditcard.csv here
│   └── .gitkeep
│
├── notebooks/                    # Jupyter notebooks (optional)
│   └── .gitkeep
│
├── models/                       # Saved models & results (auto-generated)
│   ├── best_model.pkl            # Best performing model
│   ├── scaler.pkl                # Fitted StandardScaler
│   └── training_results.json     # All evaluation metrics
│
├── assets/                       # EDA plots & charts (auto-generated)
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── time_distribution.png
│   ├── correlation_heatmap.png
│   ├── top_features.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── model_comparison.png
│
├── src/                          # Source code modules
│   ├── __init__.py               # Package initializer
│   ├── utils.py                  # Utility functions
│   ├── data_preprocessing.py     # Data loading, cleaning, scaling, SMOTE
│   ├── eda.py                    # EDA visualizations
│   └── model_training.py         # Model training & evaluation
│
├── app.py                        # Streamlit web dashboard
├── train.py                      # Main training pipeline script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── PROJECT_REPORT.md             # University project report
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9 or higher** installed on your system
- **pip** package manager
- **Git** (optional, for cloning)

### Step 1: Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd CreditCardFraudDetection

# Or simply download and extract the ZIP
```

### Step 2: Download the Dataset

1. Go to [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Click **Download** (you'll need a free Kaggle account)
3. Extract and place `creditcard.csv` in the `dataset/` folder

```
dataset/
└── creditcard.csv    ← Place here
```

### Step 3: Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install packages
pip install -r requirements.txt
```

### Step 4: Train the Models

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Generate EDA plots in `assets/`
- Train 4 ML models
- Save the best model to `models/`
- Print a comparison table

### Step 5: Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` 🎉

---

## 📖 Usage

### Training Pipeline

```bash
python train.py
```

**Output:**
- `models/best_model.pkl` — Trained model (Pickle)
- `models/scaler.pkl` — Feature scaler
- `models/training_results.json` — All metrics
- `assets/*.png` — EDA and evaluation plots

### Streamlit Dashboard

```bash
streamlit run app.py
```

**Dashboard Pages:**
1. **🏠 Home** — Project overview, summary cards, EDA visualizations
2. **🔍 Predict** — Enter transaction details, get fraud prediction with confidence
3. **📊 Performance** — Model comparison table, radar charts, confusion matrices
4. **ℹ️ About** — Project information and tech stack

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.97 | ~0.06 | ~0.92 | ~0.11 | ~0.97 |
| Decision Tree | ~0.99 | ~0.72 | ~0.76 | ~0.74 | ~0.88 |
| Random Forest | ~0.99 | ~0.93 | ~0.80 | ~0.86 | ~0.96 |
| **XGBoost** | **~0.99** | **~0.88** | **~0.82** | **~0.85** | **~0.97** |

> **Note:** Exact values will vary. The best model is selected automatically based on F1 Score. Results are approximate and depend on the SMOTE resampling.

---

## 📸 Screenshots

After running `train.py`, the following visualizations are generated in `assets/`:

- **Class Distribution** — Bar chart + pie chart showing severe class imbalance
- **Amount Distribution** — Histogram comparison of fraud vs legitimate amounts
- **Correlation Heatmap** — Feature correlation matrix
- **Confusion Matrices** — Side-by-side for all 4 models
- **ROC Curves** — Comparative ROC with AUC scores

---

## 📄 License

This project is developed for educational purposes as a university Data Science semester project. Feel free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- **Dataset**: [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Libraries**: Scikit-learn, XGBoost, Streamlit, Plotly
- **Inspiration**: Real-world fraud detection systems in financial institutions

---

<div align="center">

**Built with ❤️ for Data Science**

</div>
