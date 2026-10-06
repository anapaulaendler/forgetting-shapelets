import numpy as np
import pandas as pd

N = 6
HORIZON_DAYS = 180
LEECH = 2
CAP = 200

COLUMNS = [
    "user_id", "card_id",
    *[f"r{k}" for k in range(N)], # rating
    *[f"i{k}" for k in range(N)], # intervalo
    "lapses_h", "n_reviews", "label"   
]


def collapse_same_day(revlog: pd.DataFrame) -> pd.DataFrame:
    ordered = revlog.sort_values("day_offset")

    return ordered.drop_duplicates(["card_id", "day_offset"], keep="first")


def build_cards(revlog: pd.DataFrame, user_id: int, n: int = N, horizon_days: int = HORIZON_DAYS, leech: int = LEECH) -> pd.DataFrame:
    last_day = revlog["day_offset"].max()

    df = collapse_same_day(revlog).copy()
    df["k"] = df.groupby("card_id").cumcount()

    df["log_interval"] = np.log1p(df.groupby("card_id")["day_offset"].diff().fillna(0))

    # dia da sexta revisão
    window_end_day = df.loc[df["k"] == n - 1].set_index("card_id")["day_offset"].rename("window_end_day")

    window_end_day = window_end_day[window_end_day + horizon_days <= last_day]

    if window_end_day.empty:
        return pd.DataFrame(columns=COLUMNS)

    df = df.join(window_end_day, on="card_id", how="inner")

    window = df[df["k"] < n]
    ratings = window.pivot(index="card_id", columns="k", values="rating").add_prefix("r")

    intervals = window.pivot(index="card_id", columns="k", values="log_interval").add_prefix("i")

    in_horizon = (df["day_offset"] > df["window_end_day"]) & (df["day_offset"] <= df["window_end_day"] + horizon_days)

    lapses = ((df["rating"] == 1) & in_horizon).groupby(df["card_id"]).sum().rename("lapses_h")
    
    n_reviews = df.groupby("card_id").size().rename("n_reviews")

    cards = pd.concat([ratings, intervals, lapses, n_reviews], axis=1).rename_axis(columns=None).reset_index()
    cards["user_id"] = user_id
    cards["label"] = cards["lapses_h"] >= leech
    
    return cards[COLUMNS]


def cap_per_user(cards: pd.DataFrame, cap: int = CAP, seed: int = 0) -> pd.DataFrame:
    shuffled = cards.sample(frac=1, random_state=seed)
    capped = shuffled.groupby("user_id").head(cap)

    return capped.sort_values(["user_id", "card_id"]).reset_index(drop=True)