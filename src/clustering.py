"""
clustering.py — Clustering models (K-means, DBSCAN).

Usage:
    from src.clustering import kmeans_grid_search
    results = kmeans_grid_search(df_prepare_clustering)
"""

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.extraction import FEATURE_COLUMNS

RANDOM_STATE = 42

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

if __name__ == "__main__":
    pass