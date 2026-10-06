import numpy as np
import pandas as pd

from fs.series import N


RATING_COLS = [f"r{k}" for k in range(N)]
INTERVAL_COLS = [f"i{k}" for k in range(N)]


def to_series(cards: pd.DataFrame) -> np.ndarray:
    """(n_cards, 2 canais, N passos)"""
    return np.stack([cards[RATING_COLS].to_numpy(float), cards[INTERVAL_COLS].to_numpy(float)], axis=1)


def window_features(cards: pd.DataFrame) -> pd.DataFrame:
    ratings = cards[RATING_COLS].to_numpy()
    intervals = cards[INTERVAL_COLS].to_numpy()

    return pd.DataFrame(
        {
            "n_lapsos": (ratings == 1).sum(axis=1),
            "nota_media": ratings.mean(axis=1),
            "ultima_nota": ratings[:, -1],
            "ultimo_intervalo": intervals[:, -1],
            "crescimento_intervalo": intervals[:,-1] - intervals[:, 1] # porque i0 é sempre 0
        },
        index = cards.index
    )