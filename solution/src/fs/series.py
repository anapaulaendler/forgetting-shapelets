import numpy as np
import pandas as pd

N = 6
HORIZON_DAYS = 180
LEECH = 2
CAP = 200


def collapse_same_day(review_log: pd.DataFrame) -> pd.DataFrame:
    ordered = review_log.sort_values("day_offset")

    return ordered.drop_duplicates(["card_id", "day_offset"], keep="first")
