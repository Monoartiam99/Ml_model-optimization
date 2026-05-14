# Heart Disease Prediction — ML Optimization & Classification 

> A complete machine learning pipeline for binary classification on heart disease data, covering preprocessing, model training, evaluation, hyperparameter tuning, and model persistence.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Datasets](#-datasets)
- [What's Inside the Notebook](#-whats-inside-the-notebook)
  - [Data Preprocessing](#1-data-preprocessing)
  - [Feature Engineering](#2-feature-engineering)
  - [Model Comparison](#3-model-comparison)
  - [Hyperparameter Tuning](#4-hyperparameter-tuning)
  - [Model Persistence](#5-model-persistence)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Learning Outcomes](#-learning-outcomes)

---

## Project Overview

This notebook explores the task of predicting the presence of heart disease using supervised machine learning. The workflow covers the full ML pipeline — from raw data cleaning through to saving a production-ready model — with a focus on understanding model behaviour, evaluation metrics, and tuning strategies.

Key goals:
- Build and compare multiple classifiers on the same dataset
- Understand the impact of feature scaling and class imbalance
- Use cross-validation and GridSearchCV for robust model selection
- Save the best model for future use

---

## Datasets

| File | Description |
|---|---|
| `heart_disease_uci.csv` | UCI Heart Disease dataset — primary dataset used for training and evaluation |
| `heart.csv` | Secondary heart dataset used for correlation analysis, scaling, SMOTE, and tuning sections |

Both datasets contain a `target` column where `0 = No Disease` and `1 = Disease`.

---

## What's Inside the Notebook

### 1. Data Preprocessing

- Load CSV using `pandas`
- Drop rows with missing values (`dropna`)
- One-hot encode categorical features using `pd.get_dummies`
- Split into train/test sets (80/20) using `train_test_split`

### 2. Feature Engineering

**Correlation Analysis**
- Compute the full correlation matrix
- Visualise using a `seaborn` heatmap (coolwarm palette)
- Automatically drop features with pairwise correlation above `0.85`

**Feature Scaling**
- Apply `StandardScaler` — zero mean, unit variance
- Apply `MinMaxScaler` — scales to [0, 1] range
- Compare raw vs scaled distributions with side-by-side histograms on the `chol` feature

**Handling Imbalanced Data (SMOTE)**
- Check class distribution before resampling
- Apply `SMOTE` (Synthetic Minority Over-sampling Technique) on scaled training data
- Visualise class counts before and after with bar charts

### 3. Model Comparison

**Models trained:**

| Model | Notes |
|---|---|
| Logistic Regression | Linear classifier, `max_iter=1000`, trained on scaled features |
| Decision Tree | Tree-based classifier, trained on raw features |
| SVM | Kernel-based (`rbf`, `linear`), trained on scaled features |

**Evaluation metrics computed for each model:**
- Accuracy, Precision, Recall, F1 Score (overall and per-class)
- Confusion matrix
- ROC curve with AUC score (interactive Plotly chart)
- Optimal decision threshold using Youden's J statistic (`argmax(TPR - FPR)`)

**ROC Curve comparison:**
- Logistic Regression vs SVM plotted on the same interactive chart
- Threshold annotations shown at every 5th point

### 4. Hyperparameter Tuning

**K-Fold Cross Validation**
- Manual `cross_val_score` across SVM kernels (`rbf`, `linear`) and C values (`1`, `10`, `20`)
- Looped comparison across all kernel/C combinations to find best average score

**GridSearchCV**
- Automated search over SVM parameter grid: `C ∈ {1, 10, 20}`, `kernel ∈ {rbf, linear}`
- Results exported to a DataFrame showing `mean_test_score` per combination
- Best params and best score extracted

**Tuned Pipelines (LR & DT)**
- Logistic Regression tuned inside a `Pipeline` (StandardScaler + LR), searching over `C` values
- Decision Tree tuned over `max_depth`, `min_samples_split`, `min_samples_leaf`
- Base vs tuned comparison table: Accuracy, Precision, Recall, F1, ROC-AUC

### 5. Model Persistence

- Best model (tuned Decision Tree) serialised using `pickle`
- Saved to `best_model_tuned_dt.pkl`
- Reloaded and accuracy verified on the test set

```python
# Load the saved model
import pickle

with open("best_model_tuned_dt.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(X_test)
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Static plotting |
| `seaborn` | Heatmaps and styled plots |
| `plotly` | Interactive ROC curve charts |
| `scikit-learn` | Models, metrics, preprocessing, tuning |
| `imbalanced-learn` | SMOTE for class imbalance |
| `pickle` | Model serialisation |

---

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/Monoartiam99/Ml_model-optimization
cd your-repo-name
```

**2. Install dependencies**

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn jupyter
```

**3. Launch the notebook**

```bash
jupyter notebook assigment_3_1.ipynb
```

> Make sure `heart.csv` and `heart_disease_uci.csv` are in the same directory as the notebook before running.

---

## Learning Outcomes

By working through this notebook you will understand how to:

- Clean and encode real-world tabular data for ML
- Train and evaluate classification models (Logistic Regression, Decision Tree, SVM)
- Interpret confusion matrices, precision, recall, F1, and ROC-AUC
- Detect and remove highly correlated features
- Apply and compare feature scaling methods
- Handle class imbalance using SMOTE
- Perform K-Fold cross validation manually and with GridSearchCV
- Build scikit-learn Pipelines for clean model tuning
- Save and reload trained models with pickle
  
## My Model Test Accuracy
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/911c9bb5-210a-4223-8a70-b6451e1a818d" />

### Screenshots of wakatime:
<img width="1919" height="905" alt="image" src="https://github.com/user-attachments/assets/09f8f789-eb8b-4598-a5d3-0b80502fbdc9" />
