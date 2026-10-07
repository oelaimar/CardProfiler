import time
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.extraction import FEATURE_COLUMNS

RANDOM_STATE = 42
TARGET_COLUMN = "target"
CLUSTERING_COLUMNS = ["cluster_kmeans", "cluster_dbscan", "est_atypique_dbscan", "cluster_final"]

def build_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2 ):
    # stratify to keet the same proportion of each class
    return train_test_split(X ,y, test_size=test_size, stratify=y)

def analyze_imbalance(y:pd.Series, max_ratio: float = 1.5, min_pct: float = 10.0) -> tuple[pd.DataFrame, float, bool]:
    counts = y.value_counts()
    table = pd.DataFrame({
        "count" : counts,
        "pct"   : (counts / counts.sum() * 100).round(1),
    })
    ratio = round(counts.max() / counts.min() , 2)
    needs_resampling = ratio > max_ratio or table["pct"].min() < min_pct

    return table, ratio, needs_resampling

def make_sampler(simpling: str) :
    samplers = {
        "none" : None,
        "over" : RandomOverSampler(random_state=RANDOM_STATE),
        "smote": SMOTE(random_state=RANDOM_STATE),
        "under": RandomUnderSampler(random_state=RANDOM_STATE),
    }
    if simpling not in samplers:
        raise ValueError(f"sampling must be one of {list(samplers)}")
    return samplers[simpling]

def build_pipeline(classifier, sampling: str = "none") -> ImbPipeline:
    steps = [("scaler", StandardScaler())]
    sampler = make_sampler(sampling)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("classifier", classifier)),
    return ImbPipeline(steps)

CLASSIFIRES = {
        "random_forest"         : RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        # "svm": SVC(probability=True, random_state=RANDOM_STATE),
        "svm"                   : CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False),
        "decision_tree"         : DecisionTreeClassifier(random_state=RANDOM_STATE),
        "logistic_regression"   : LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }

def train_model(X_train: pd.DataFrame, y_train: pd.Series, sampling: str = "none" ) -> tuple[dict, pd.DataFrame]:
    pipelines = {}
    times = []
    for name, classifier in CLASSIFIRES.items():
        pipeline = build_pipeline(classifier, sampling=sampling)

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        times.append({"model" : name, "train_time_s" : round(elapsed,3)})
        pipelines[name] = pipeline
    return pipelines, pd.DataFrame(times).set_index("model")

def evaluate_models(pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series, train_times: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        rows.append({
            "model"           : name,
            "accuracy"        : accuracy_score(y_test, y_pred),
            "precision_macro" : precision_score(y_test, y_pred, average="macro"),
            "recall_macro"    : recall_score(y_test, y_pred, average="macro"),
            "f1_macro"        : f1_score(y_test, y_pred, average="macro")
        })
    table = pd.DataFrame(rows).set_index("model").join(train_times)
    return table.sort_values("f1_macro", ascending=False).round(4)

def plot_confusion_matrix(pipeline, X_test: pd.DataFrame, y_test: pd.Series, title : str = "", ax=None):
    labels = pipeline.classes_
    cm = confusion_matrix(y_test, pipeline.predict(X_test), labels=labels)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted segment")
    ax.set_ylabel("True segment")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    return ax

def plot_all_confusion_matrices(pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, pipeline) in zip(axes.ravel(), pipelines.items()):
        plot_confusion_matrix(pipeline, X_test, y_test, title=f"Confusion matrix — {name}", ax=ax)
    fig.tight_layout()
    return fig

def plot_f1_comparison(comparison: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    data = comparison["f1_macro"].sort_values()
    bars = ax.barh(data.index, data.values,
                   color=["C2" if i == len(data) - 1 else "C0" for i in range(len(data))])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlim(data.min() - 0.05, 1.0)
    ax.set_xlabel("Macro F1-score (test set)")
    ax.set_title("Model comparison — macro F1")
    return ax

def cross_validate_models(X_train: pd.DataFrame, y_train: pd.Series,
                          sampling: str = "none", n_splits: int = 5,
                          scoring: str = "f1_macro")-> pd.DataFrame:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, classifier in CLASSIFIRES.items():
        pipeline = build_pipeline(classifier, sampling=sampling)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        for fold, score in enumerate(scores, start=1):
            rows.append({"model": name, "fold": fold, scoring: score})
    return pd.DataFrame(rows)

def plot_cv_boxplot(cv_scores: pd.DataFrame, scoring: str = "f1_macro", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    order = cv_scores.groupby("model")[scoring].median().sort_values(ascending=False).index
    sns.boxplot(data=cv_scores, x=scoring, y="model", order=order, ax=ax)
    sns.stripplot(data=cv_scores, x=scoring, y="model", order=order,
                  color="black", size=5, ax=ax)
    ax.set_xlabel(f"{scoring} (each dot = one fold)")
    ax.set_ylabel("")
    ax.set_title("Cross-validation — model stability")
    return ax

param_grids = {
    "random_forest": {
        "classifier__n_estimators": [100, 200, 400],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 5],
    },
    "svm": {
        "classifier__estimator__C": [0.1, 1, 10, 100],
        "classifier__estimator__gamma": ["scale", 0.01, 0.1, 1],
    },
    "decision_tree": {
        "classifier__max_depth": [3, 5, 8, 12, None],
        "classifier__min_samples_leaf": [1, 5, 10, 20],
        "classifier__criterion": ["gini", "entropy"],
    },
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
    },
}

def tune_models(X_train: pd.DataFrame, y_train: pd.Series, sampling: str = "none",
                search: str = "grid", n_iter: int = 20, n_splits: int = 5,
                scoring: str = "f1_macro")-> tuple[dict, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    grids = param_grids
    best_pipelines, rows = {}, []

    for name, classifier in CLASSIFIRES.items():
        pipeline = build_pipeline(classifier, sampling=sampling)
        grid = grids[name]
        if search == "grid":
            searcher = GridSearchCV(pipeline, grid, cv=cv, scoring=scoring, n_jobs=-1)
        elif search == "random":
            n_combinations = len(ParameterGrid(grid))
            searcher = RandomizedSearchCV(
                pipeline, grid, n_iter=min(n_iter, n_combinations), cv=cv, scoring=scoring, n_jobs=-1, random_state=RANDOM_STATE
            )
        else:
            raise ValueError("search must be 'grid' or 'random'")

        start = time.perf_counter()
        searcher.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        best_pipelines[name] = searcher.best_estimator_
        rows.append({
            "model": name,
            "best_cv_f1_macro": round(searcher.best_score_, 4),
            "best_params": searcher.best_params_,
            "search_time_s": round(elapsed, 1),
        })

    summary = pd.DataFrame(rows).set_index("model").sort_values("best_cv_f1_macro", ascending=False)

    return best_pipelines, summary

def select_best_model(comparison: pd.DataFrame, pipelines: dict, metric: str = "f1_macro") -> tuple[str, object]:
    best_name = comparison[metric].idxmax()
    return best_name, pipelines[best_name]

def get_feature_importance(pipeline, X_test=None, y_test=None) -> pd.Series:
    model = pipeline.named_steps["classifier"]

    if hasattr(model, "feature_importances_"): # Random Forest, Decision Tree
        values, method = model.feature_importances_, "feature_importances_"
    elif hasattr(model, "coef_"):  # Logistic Regression
        values, method = np.abs(model.coef_).mean(axis=0), "mean |coefficient|"
    else:
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test are needed for permutation importance")
        result = permutation_importance(pipeline, X_test, y_test, scoring="f1_macro",
                                        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        values, method = result.importances_mean, "permutation importance"
    return pd.Series(values, index=FEATURE_COLUMNS, name=method).sort_values()

def plot_feature_importance(importance: pd.Series, model_name: str = "", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(importance.index, importance.values, color="C0")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel(importance.name)
    ax.set_title(f"Feature importance — {model_name}")
    return ax