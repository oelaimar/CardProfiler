"""
clustering.py — Clustering models (K-means, DBSCAN).

Usage:
    from src.clustering import kmeans_grid_search
    results = kmeans_grid_search(df_prepare_clustering)
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import centroid
from seaborn import matrix

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.extraction import FEATURE_COLUMNS

RANDOM_STATE = 42
METHOD_COLUMNS = {"kmeans": "cluster_kmeans", "dbscan": "cluster_dbscan"}

# KMeans
def apply_pca(df_scaled : pd.DataFrame, n_components:int) -> tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    value = pca.fit_transform(df_scaled[FEATURE_COLUMNS])
    columns = [f"PC{i + 1}" for i in range(n_components)]
    df_pca = pd.DataFrame(value, columns=columns, index=df_scaled.index)
    return df_pca, pca

def kmeans_grid_search(
        df_scaled: pd.DataFrame,
        components_range = range(2, 6),
        k_range = range(2, 8),
        silhouette_sample: int | None = None,
) -> pd.DataFrame:
    result = []
    for n_comp in components_range:
        df_pca , pca = apply_pca(df_scaled, n_comp)
        explained = pca.explained_variance_ratio_.sum()
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            labels = kmeans.fit_predict(df_pca)

            silhouette = silhouette_score(df_pca, labels, sample_size=silhouette_sample, random_state=RANDOM_STATE)

            sizes = np.bincount(labels) / len(labels)

            result.append({
                "n_components" : n_comp,
                "k": k,
                "explained_variance" : round(explained, 3),
                "inertia": round(kmeans.inertia_, 1),
                "silhouette": round(silhouette, 4),
                "min_cluster_pct": round(sizes.min() * 100, 1),
                "max_cluster_pct": round(sizes.max() * 100, 1),
            })
    return (
        pd.DataFrame(result).sort_values("silhouette", ascending=False).reset_index(drop=True)
    )

def check_cluster_balance(labels, min_pct: float = 10.0, max_pct : float = 90.0) -> tuple[pd.DataFrame, bool]:
    counts = pd.Series(labels).value_counts().sort_index()
    sizes = pd.DataFrame({
        "count" : counts,
        "pct" : (counts / counts.sum() * 100).round(1),
    })
    sizes.index.name = "cluster"

    is_balanced = bool(sizes["pct"].between(min_pct, max_pct).all())
    return sizes, is_balanced

def fit_final_kmeans(df_scaled: pd.DataFrame, n_components: int, k: int) -> tuple[np.ndarray, KMeans, PCA, pd.DataFrame]:
    df_pca, pca = apply_pca(df_scaled, n_components)
    k_means = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = k_means.fit_predict(df_pca)
    return labels, k_means, pca, df_pca

def attach_labels(df_clean: pd.DataFrame, labels, column: str) -> pd.DataFrame:
    if len(labels) != len(df_clean):
        raise ValueError(f"Length mismatch: {len(labels)} labels for {len(df_clean)} clients")
    df = df_clean.copy()
    df[column] = np.asarray(labels)
    return df

# DBSCAN

def dbscan_grid_search(
        df_scaled: pd.DataFrame,
        components_range = range(2, 6),
        eps_values=(0.3, 0.5, 0.7, 1.0, 1.3, 1.6),
        min_samples_values = (5, 10, 20, 50),
        silhouette_sample : int | None = 3000,
) -> pd.DataFrame:
    results = []

    for n_comp in components_range:
        df_pca, _ = apply_pca(df_scaled, n_comp)
        for eps in eps_values:
            for min_samples in min_samples_values:
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(df_pca)
                is_noise = labels == -1
                noise_pct = is_noise.mean() * 100
                X_clustered = df_pca[~is_noise]
                cluster_labels = labels[~is_noise]

                n_clusters = len(np.unique(cluster_labels))

                silhouette = np.nan

                if n_clusters >= 2:
                    sample = None
                    if silhouette_sample is not None:
                        sample = min(silhouette_sample, len(cluster_labels))
                    try:
                        silhouette = silhouette_score(
                            X_clustered, cluster_labels,
                            sample_size=sample, random_state=RANDOM_STATE,
                        )
                    except ValueError:
                        silhouette = np.nan

                if n_clusters > 0:
                    sizes = np.bincount(cluster_labels) / len(labels) * 100
                    min_pct, max_pct = sizes.min(), sizes.max()
                else:
                    min_pct = max_pct = 0
                results.append({
                    "n_components": n_comp,
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "noise_pct": round(noise_pct, 1),
                    "silhouette": round(silhouette, 4) if not np.isnan(silhouette) else np.nan,
                    "min_cluster_pct": round(min_pct, 1),
                    "max_cluster_pct": round(max_pct, 1),
                })
    return (
        pd.DataFrame(results)
        .sort_values("silhouette", ascending=False, na_position="last")
        .reset_index(drop=True))

def fit_final_dbscan(df_scaled: pd.DataFrame, n_components: int, eps: float, min_samples:int) -> tuple[np.ndarray, DBSCAN, pd.DataFrame]:
    df_pca, _ = apply_pca(df_scaled, n_components)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df_pca)

    return labels, dbscan, df_pca

def clustering_metrics(df_pca: pd.DataFrame, labels, method :str ) -> dict:
    labels = np.asarray(labels)
    clustered = labels != -1
    X, y = df_pca[clustered], labels[clustered]
    n_clusters = len(np.unique(y))
    metrics = {
        "method": method,
        "n_clusters": n_clusters,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "noise_pct": round((~clustered).mean() * 100, 1),
        "coverage_pct": round(clustered.mean() * 100, 1),
    }
    if n_clusters >= 2:
        metrics["silhouette"] = round(silhouette_score(X, y), 4)
        metrics["davies_bouldin"] = round(davies_bouldin_score(X, y), 4)
        metrics["calinski_harabasz"] = round(calinski_harabasz_score(X, y), 1)
    return metrics

def compare_methods(df_pca_kmeans : pd.DataFrame, labels_kmeans , df_pca_dbscan: pd.DataFrame, labels_dbscan) -> pd.DataFrame:
    rows = []
    for method, df_pca, labels in [
        ("kmeans", df_pca_kmeans, labels_kmeans),
        ("dbscan", df_pca_dbscan, labels_dbscan),
        ]:
        metrics = clustering_metrics(df_pca, labels, method)
        labels = np.asarray(labels)
        sizes, is_balanced = check_cluster_balance(labels[labels != -1])
        metrics["min_cluster_pct"] = sizes["pct"].min()
        metrics["max_cluster_pct"] = sizes["pct"].max()
        metrics["is_balanced"] = is_balanced
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("method")

def choose_method(
        comparison: pd.DataFrame,
        silhouette_margin: float = 0.05,
        max_noise_pct: float = 5.0,) -> tuple[str, str]:
    km, db = comparison.loc["kmeans"], comparison.loc["dbscan"]

    dbscan_wins = (
        pd.notna(db["silhouette"])
        and db["silhouette"] >= km["silhouette"] + silhouette_margin
        and db["noise_pct"] < max_noise_pct
        and db["is_balanced"]
    )
    if dbscan_wins:
        return "dbscan", (
            f"* DBSCAN retained: silhouette {db['silhouette']} vs {km['silhouette']} \n" 
            f"* (margin >= {silhouette_margin}), noise {db['noise_pct']}% < {max_noise_pct}%, \n" 
            f"* balanced clusters.\n"
        )
    return "kmeans", (
        f"* K-means retained: it classifies 100% of clients (needed for target). \n" 
        f"* DBSCAN silhouette = {db['silhouette']}, noise = {db['noise_pct']}%, \n" 
        f"* balanced = {db['is_balanced']} — not clearly better.\n"
    )

def set_cluster_final(df_clean: pd.DataFrame, method: str) -> pd.DataFrame:
    if method not in METHOD_COLUMNS:
        raise ValueError(f"method must be one of {list(METHOD_COLUMNS)}")
    source = METHOD_COLUMNS[method]
    if source not in df_clean.columns:
        raise KeyError(f"Column '{source}' not found — attach labels first")

    df = df_clean.copy()
    df["cluster_final"] = df[source]

    if df["cluster_final"].isna().any():
        raise ValueError("cluster_final contains missing values")
    if (df["cluster_final"] == -1).any():
        raise ValueError("cluster_final contains noise (-1): not usable as target")
    return df

def plot_clusters_2d(df_scaled: pd.DataFrame, cluster_labels, ax=None):
    df_2d, pca = apply_pca(df_scaled, 2)
    labels = np.asarray(cluster_labels)

    centroids = df_2d.groupby(labels).mean()

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        ax.scatter(
            df_2d.loc[mask, "PC1"], df_2d.loc[mask, "PC2"],
            s=8, alpha=0.4,
            label=f"Cluster {cluster} ({mask.mean() * 100:.1f}%)"
        )

    ax.scatter(
        centroids["PC1"], centroids["PC2"],
        marker="X", s=250, c="black", edgecolors="white", linewidths=0.5,
        label="Centroids",
    )
    for cluster, row in centroids.iterrows():
        ax.annotate(str(cluster), (row["PC1"], row["PC2"]), xytext=(8, 8), textcoords="offset points", fontweight="bold")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} of variance)")
    ax.set_title("Customer segments (cluster_final) in PCA space")
    ax.legend(markerscale=2)
    return ax, centroids

def cluster_profile_standardized(df_scaled: pd.DataFrame, cluster_labels) -> pd.DataFrame:
    return (
        df_scaled[FEATURE_COLUMNS].groupby(np.asarray(cluster_labels)).mean().rename_axis("cluster")
    )
def plot_cluster_profile_heatmap(profile: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 0.8 * len(profile) + 2))
        sns.heatmap(
            profile, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0,
            linewidths=0.5, cbar_kws={"label": "Mean (in standard deviations)"},
            ax=ax,
        )
    ax.set_title("Cluster profiles — standardized means")
    ax.set_xlabel("")
    ax.set_ylabel("Cluster")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return ax

if __name__ == "__main__":
    pass