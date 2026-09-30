"""
clustering.py — Clustering models (K-means, DBSCAN).

Usage:
    from src.clustering import kmeans_grid_search
    results = kmeans_grid_search(df_prepare_clustering)
"""

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.extraction import FEATURE_COLUMNS

RANDOM_STATE = 42

def apply_pca(df_scaled : pd.DataFrame, n_components:int) -> tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    value = pca.fit_transform(df_scaled[FEATURE_COLUMNS])
    columns = [f"pc{i + 1}" for i in range(n_components)]
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
