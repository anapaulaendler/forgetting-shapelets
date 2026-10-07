from datetime import datetime, timedelta, timezone

from fsrs import Card, Rating, Scheduler
import numpy as np
import pandas as pd

from fs.series import N


RATING_COLS = [f"r{k}" for k in range(N)]
INTERVAL_COLS = [f"i{k}" for k in range(N)]
BASE_DATE = datetime(2000, 1, 1, tzinfo=timezone.utc)


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


def review_days(cards: pd.DataFrame) -> np.ndarray:
    gaps = np.rint(np.expm1(cards[INTERVAL_COLS].to_numpy(float))).astype(int)

    return gaps.cumsum(axis=1)


def fsrs_state(cards: pd.DataFrame) -> pd.DataFrame:
    scheduler = Scheduler(enable_fuzzing=False)
    rows = []

    for k, (days, ratings) in enumerate(zip(review_days(cards), cards[RATING_COLS].to_numpy(int))):
        card = Card(card_id=k)

        for day, r in zip(days, ratings):
            card, _ = scheduler.review_card(card, Rating(int(r)), review_datetime=BASE_DATE + timedelta(days=int(day)))

        rows.append((card.stability, card.difficulty))

    return pd.DataFrame(rows, columns=["stability", "dificulty"], index=cards.index)