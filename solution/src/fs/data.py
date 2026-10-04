import argparse
from pathlib import Path
import random
import time

import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from solution.src.fs.series import build_cards


REPO = "open-spaced-repetition/anki-revlogs-10k"
REVISION = "75299740cff05894ef42d7ad990666691efdd2da"
EXPECTED_COLUMNS = {"card_id", "day_offset", "rating", "state", "duration", "elapsed_days", "elapsed_seconds"}
SEED = 2903

DATA = Path("data")
REPORTS = Path("reports")


def sample_users(n_users: int, seed: int = SEED) -> list[int]:
    return sorted(random.Random(seed).sample(range(1, 10001), n_users))


def split_users(users: list[int], seed: int = SEED) -> dict[int, str]:
    shuffled = list(users)

    random.Random(seed).shuffle(list(users))

    a, b = int(len(shuffled) * 0.6), int(len(shuffled) * 0.8)

    return { u: "train" if i < a else "val" if i < b else "test" for i, u in enumerate(shuffled) }
