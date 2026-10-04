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


def load_revlog(user_id: int) -> pd.DataFrame:
    path = hf_hub_download(REPO, f"revlogs/user_id={user_id}/data.parquet", repo_type="dataset", revision=REVISION)

    df = pd.read_parquet(path)

    return df

def write_report(eligible: pd.DataFrame, cards: pd.DataFrame, users: list[int], total_reviews: int, seconds: float) -> None:
    figs = REPORTS / "etapa0"
    figs.mkdir(parents=True, exist_ok=True)

    train = cards[cards["split"] == "train"]
    per_user = eligible.groupby("user_id").size().reindex(users, fill_value=0)
    top2 = per_user.nlargest(2).sum() / max(len(eligible), 1)

    fig, ax = plt.subplots()
    ax.hist(cards["n_reviews"], bins=50, log=True)
    ax.set(xlabel="revisoes por card (depois de colapsar o mesmo dia)", ylabel="cards (escala log)")
    fig.savefig(figs / "n_reviews.png")
    plt.close(fig)

    normal, prob = train[~train["label"], train[train["label"]]]
    examples = pd.concat([normal.sample(min(2, len(normal)), random_state=0), prob.sample(min(1, len(prob)), random_state=0)])

    fig, axes = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
    for _, row in examples.iterrows():
        tag = "problematico" if row["label"] else "normal"

        axes[0].plot(range(N), [row[f"r{k}"] for k in range(N)], marker="o", label=tag)
        axes[1].plot(range(N), [row[f"i{k}"] for k in range(N)], marker="o")

    axes[0].set(ylabel="nota (1 = esqueceu)", yticks=[1, 2, 3, 4])
    axes[0].legend()

    axes[1].set(ylabel="log(1 + dias)", xlabel="revisao da janela")

    fig.savefig(figs / "series_exemplo.png", dpi=150)
    plt.close(fig)

    splits = cards.groupby("split")["user_id"].unique().to_dict()
    prev = { l: (train["lapses_h"] >= l).mean() for l in (1, 2, 3)}

    liness = [
        "etapa 0: base comum",
        "",
        f"gerado com `uv run python -m fs.data --n-users {len(users)}` em {seconds / 60:.1f} min"
        f"dataset `{REPO}` @ `{REVISION[:12]}` . seed {SEED} . N = {N}, H = {H} dias, L = {L}, teto = {CAP}.",
        "",
        f"usuarios por split (com algum card) => treino {splits.get('train', 0)} . val {splits.get('val', 0)} . teste {splits.get('test', 0)}",
        f"usuarios sem nenhum card elegivel => {(per_user == 0).sum()}",
        f"revisoes brutas baixadas => {total_reviews:,}",
        f"cards elegiveis antes do teto => {len(eligible):,} (top-2 usuarios: {top2:.0%})",
        f"cards elegiveis por usuario => mediana {per_user.median():.0f} . p90 {per_user.quantile(0.9):.0f} . max {per_user.max()} |",
        f"cards apos o teto => {len(cards):,}",
        f"prevalencia de card problematico no treino (L = {L}) => {prev[L]:.1%}",
        f"robustez no treino: L = 1 . L = 3 => {prev[1]:.1%} . {prev[3]:.1%}",
        "",
        "![revisões por card](etapa0/n_reviews.png)",
        "",
        "![séries de exemplo](etapa0/series_exemplo.png)",
        "",
    ]

    (REPORTS / "etapa0.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=500)
    args = parser.parse_args()

    start = time.time()
    users = sample_users(args.n_users)
    split = split_users(users)
    parts, total_reviews = [], 0
    
    for i, u in enumerate(users, 1):
        revlog = load_revlog(u)
        total_reviews += len(revlog)
        parts.append(build_cards(revlog, u))
        print(f"[{i}/{len(users)}] user {u}: {len(parts[-1])} cards elegiveis", flush=True)

    eligible = pd.concat([p for p in parts if len(p)], ignore_index=True)
    cards = cap_per_user(eligible)
    cards["split"] = cards["user_id"].map(split)

    DATA.mkdir(exist_ok=True)
    
    cards.to_parquet(DATA / "cards.parquet", index=False)
    
    write_report(eligible, cards, users, total_reviews, time.time() - start)
    
    print(f"ok: {len(cards)} cards em {DATA / 'cards.parquet'} · relatório em {REPORTS / 'etapa0.md'}")


if __name__ == "__main__":
    main()