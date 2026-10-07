import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

N_BOOT = 1000


def metrics(y, p) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    both = 0 < y.sum() < len(y)

    return {
        "pr_auc": average_precision_score(y, p) if both else np.nan,
        "roc_auc": roc_auc_score(y, p) if both else np.nan,
        "brier": brier_score_loss(y, p),
    } # type: ignore


def user_groups(users) -> list[np.ndarray]:
    users = np.asarray(users)
    
    return list(pd.Series(users).groupby(users).indices.values()) # type: ignore


def resample_users(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    drawn = rng.integers(0, len(groups), len(groups))

    return np.concatenate([groups[g] for g in drawn])


def bootstrap(users, y, preds: dict[str, np.ndarray], n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    y, groups, rng = np.asarray(y), user_groups(users), np.random.default_rng(seed)
    rows = []

    for b in range(n_boot):
        idx = resample_users(groups, rng)
        for name, p in preds.items():

            for metric, value in metrics(y[idx], np.asarray(p)[idx]).items():
                rows.append((b, name, metric, value))

    return pd.DataFrame(rows, columns=["boot", "model", "metric", "value"])


def ci_table(y, preds: dict[str, np.ndarray], samples: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for name, p in preds.items():
        for metric, point in metrics(y, p).items():
            v = samples.loc[(samples.model == name) & (samples.metric == metric), "value"]
            rows.append((name, metric, point, np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)))

    return pd.DataFrame(rows, columns=["model", "metric", "point", "lo", "hi"])


def diff_ci(samples: pd.DataFrame, a: str, b: str, metric: str = "pr_auc") -> tuple[float, float]:
    wide = samples[samples.metric == metric].pivot(index="boot", columns="model", values="value")
    d = (wide[a] - wide[b]).to_numpy()

    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "vence"

    if hi < 0:
        return "perde"

    return "empate"