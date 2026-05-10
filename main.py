import io
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Heart Disease ML", layout="wide")


#  helpers 
@st.cache_data
def load(file):
    df = pd.read_csv(file).dropna()
    df = pd.get_dummies(df, drop_first=True)
    return df

def get_metrics(name, model, X_t, y_t):
    y_pred = model.predict(X_t)
    y_prob = model.predict_proba(X_t)[:, 1]
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_t, y_pred),  4),
        "Precision": round(precision_score(y_t, y_pred), 4),
        "Recall":    round(recall_score(y_t, y_pred),    4),
        "F1":        round(f1_score(y_t, y_pred),        4),
        "ROC-AUC":   round(roc_auc_score(y_t, y_prob),   4),
    }


# sidebar 
st.sidebar.title(" Settings")
uploaded  = st.sidebar.file_uploader("Upload heart.csv", type=["csv"])
test_size = st.sidebar.slider("Test size", 0.10, 0.40, 0.20, 0.05)
seed      = st.sidebar.number_input("Random seed", value=42)
page      = st.sidebar.radio("Go to", [
    " Data Overview",
    " Model Metrics",
    " ROC Curves",
    " GridSearchCV",
    " Save Model",
])

st.title(" Heart Disease ML Dashboard")

if uploaded is None:
    st.info("Upload a CSV file in the sidebar to get started.")
    st.stop()

df = load(uploaded)

if "target" not in df.columns:
    st.error("Dataset must have a **target** column (0 / 1).")
    st.stop()

X, y = df.drop("target", axis=1), df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed, stratify=y
)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# Data Overview 
if page == " Data Overview":
    st.header(" Data Overview")

    c1, c2 = st.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=True,
                fmt=".2f", linewidths=0.4, cbar=False, ax=ax)
    st.pyplot(fig)


#  Model Metrics 
elif page == " Model Metrics":
    st.header(" Model Metrics")

    with st.spinner("Training…"):
        lr = LogisticRegression(max_iter=1000, random_state=seed)
        lr.fit(X_train_sc, y_train)

        dt = DecisionTreeClassifier(random_state=seed)
        dt.fit(X_train, y_train)

    results = pd.DataFrame([
        get_metrics("Logistic Regression", lr, X_test_sc, y_test),
        get_metrics("Decision Tree",       dt, X_test,    y_test),
    ]).set_index("Model")
    st.dataframe(results, use_container_width=True)

    st.subheader("Confusion Matrices")
    c1, c2 = st.columns(2)
    for col, name, model, X_t in [
        (c1, "Logistic Regression", lr, X_test_sc),
        (c2, "Decision Tree",       dt, X_test),
    ]:
        cm  = confusion_matrix(y_test, model.predict(X_t))
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["Actual 0", "Actual 1"])
        ax.set_title(name)
        col.pyplot(fig)


#  ROC Curves 
elif page == " ROC Curves":
    st.header(" ROC Curves")

    with st.spinner("Fitting models…"):
        lr = LogisticRegression(max_iter=1000, random_state=seed)
        lr.fit(X_train_sc, y_train)
        dt = DecisionTreeClassifier(random_state=seed)
        dt.fit(X_train, y_train)

    fig = go.Figure()
    for name, model, X_t in [
        ("Logistic Regression", lr, X_test_sc),
        ("Decision Tree",       dt, X_test),
    ]:
        prob = model.predict_proba(X_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{name} (AUC={auc:.2f})"))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Random", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="FPR", yaxis_title="TPR",
                      title="ROC Curve", height=500)
    st.plotly_chart(fig, use_container_width=True)


#  GridSearchCV 

elif page == " GridSearchCV":
    st.header(" GridSearchCV — Decision Tree")

    with st.spinner("Running search (this takes a moment)…"):
        grid = GridSearchCV(
            DecisionTreeClassifier(random_state=seed),
            {"max_depth": [3, 5, 7, None],
             "min_samples_split": [2, 5, 10]},
            cv=5, scoring="roc_auc", n_jobs=-1,
        )
        grid.fit(X_train, y_train)

    st.success(f"Best params: `{grid.best_params_}`  |  Best AUC: `{grid.best_score_:.4f}`")

    cv_df = (pd.DataFrame(grid.cv_results_)
             [["param_max_depth", "param_min_samples_split",
               "mean_test_score", "rank_test_score"]]
             .sort_values("rank_test_score"))
    st.dataframe(cv_df, use_container_width=True)


#  Save Model 

elif page == " Save Model":
    st.header(" Save Model")

    with st.spinner("Training best model…"):
        grid = GridSearchCV(
            DecisionTreeClassifier(random_state=seed),
            {"max_depth": [3, 5, 7, None], "min_samples_split": [2, 5, 10]},
            cv=5, scoring="roc_auc", n_jobs=-1,
        )
        grid.fit(X_train, y_train)

    acc = accuracy_score(y_test, grid.predict(X_test))
    st.metric("Test Accuracy", f"{acc:.4f}")

    buf = io.BytesIO()
    pickle.dump(grid, buf)
    buf.seek(0)
    st.download_button(" Download model.pkl", buf,
                       file_name="model.pkl",
                       mime="application/octet-stream")