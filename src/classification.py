import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

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
