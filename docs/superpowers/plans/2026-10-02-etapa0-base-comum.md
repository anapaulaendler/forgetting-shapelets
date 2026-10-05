# Etapa 0 — Base comum · Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Preferência da Ana:** executar **uma task por vez** e esperar aprovação antes da próxima
> (a menos que a mensagem traga `--skip-validations`).

**Goal:** baixar um subconjunto fixo de 500 usuários do `anki-revlogs-10k`, transformar cada revlog
em uma linha por card elegível (janela de 6 revisões + rótulo de 180 dias), aplicar teto e split por
usuário, e publicar `reports/etapa0.md` com os números e figuras da base.

**Architecture:** `src/fs/series.py` tem só funções puras sobre DataFrames (testáveis sem rede):
colapsar o mesmo dia, montar cards, aplicar teto. `src/fs/data.py` cuida de I/O: sorteio e split de
usuários, download do HF, validação de schema, gravação de `data/cards.parquet` e do relatório.

**Tech Stack:** Python 3.12 (via uv), pandas, pyarrow, huggingface_hub, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-forgetting-shapelets-design.md` (§2, §4 Definições, §7, §8)

## Global Constraints

- **N = 6** revisões na janela; **H = 180** dias de horizonte; **L = 2** lapsos; **teto 200** cards por usuário.
- Lapso = nota 1. Revisões no mesmo dia colapsadas por `(card_id, day_offset)`, mantendo a primeira.
- Horizonte do rótulo: revisões com `day_offset` em **(dia da 6ª, dia da 6ª + 180]**.
- Elegível: card com ≥ 6 revisões **e** `max(day_offset)` do **usuário** ≥ dia da 6ª + 180.
- **500 usuários**, seed fixa; dataset `open-spaced-repetition/anki-revlogs-10k`, revisão
  `75299740cff05894ef42d7ad990666691efdd2da`.
- Split por usuário, disjunto. **Decisão deste plano** (a spec não fixa proporções): 60/20/20
  treino/val/teste.
- Prevalência reportada **só no treino**.
- Licença proíbe redistribuir: **`data/` nunca entra no git**; `reports/` só agregados e figuras.
- Etapa roda em **< 30 min**. Se não couber, reduzir usuários (não otimizar código).
- `data.py` aborta com mensagem clara se o schema não tiver as colunas esperadas.

## Review Focus

1. **Sem acesso ao dataset gated** (sem token ou termos não aceitos) → mensagem dizendo o que fazer,
   não um traceback de 401. *Verificado manualmente na Task 4, Step 5.*
2. **Schema do parquet mudou** → aborta nomeando as colunas faltando. *Teste na Task 3.*
3. **Usuário sem nenhum card elegível** → contribui zero cards, sem quebrar o `concat`.
   *Teste na Task 2 (retorno vazio com as colunas certas).*
4. **Revisões fora de ordem ou várias no mesmo dia** → janela e intervalos calculados sobre dias
   distintos, em ordem. *Testes nas Tasks 1 e 2.*
5. **Rodar de novo** → mesmos usuários, mesmo split, mesmo teto. *Testes de determinismo na Task 3.*

---

### Task 1: Projeto + colapso de revisões no mesmo dia

**Files:**
- Create: `pyproject.toml`, `.python-version` (via `uv python pin`), `src/fs/__init__.py`, `src/fs/series.py`
- Modify: `.gitignore` (adicionar `data/`)
- Test: `tests/test_series.py`

**Interfaces:**
- Produces: `fs.series.N = 6`, `H = 180`, `L = 2`, `CAP = 200`;
  `collapse_same_day(revlog: pd.DataFrame) -> pd.DataFrame`.

- [ ] **Step 1: Instalar uv** (não está instalado nesta máquina; o linuxbrew está no PATH)

Run: `brew install uv && uv --version`
Expected: imprime a versão do uv.

- [ ] **Step 2: Criar `pyproject.toml`**

```toml
[project]
name = "forgetting-shapelets"
version = "0.1.0"
description = "Shapelets para prever cards problemáticos e esquecimento em revisão espaçada"
requires-python = ">=3.12,<3.14"
dependencies = [
    "pandas>=2.2",
    "pyarrow>=17",
    "huggingface_hub>=0.25",
    "matplotlib>=3.9",
]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/fs"]
```

`<3.14` porque o Python do sistema é 3.14 e o aeon (Etapa 1) pode não ter wheel para ele ainda.
aeon, fsrs e scikit-learn entram no plano da Etapa 1, quando forem usados.

- [ ] **Step 3: Fixar Python, criar pacote vazio, ignorar `data/`**

```bash
uv python pin 3.12
mkdir -p src/fs tests
touch src/fs/__init__.py
printf '\n# dado do anki-revlogs-10k — licença proíbe redistribuir\ndata/\n' >> .gitignore
uv sync
```

Expected: `uv sync` cria `.venv/` e `uv.lock` sem erro.

- [ ] **Step 4: Escrever os testes que falham**

`tests/test_series.py`:

```python
import numpy as np
import pandas as pd
import pytest

from fs.series import collapse_same_day


def revlog(rows):
    """rows: (card_id, day_offset, rating). Demais colunas do schema com valor neutro."""
    df = pd.DataFrame(rows, columns=["card_id", "day_offset", "rating"])
    return df.assign(state=2, duration=1000, elapsed_days=0, elapsed_seconds=0)


def card(card_id, days, ratings):
    return list(zip([card_id] * len(days), days, ratings))


SIX_DAYS = [0, 1, 3, 7, 15, 30]  # 6ª revisão no dia 30 → horizonte (30, 210]


def test_collapse_keeps_first_review_of_the_day():
    out = collapse_same_day(revlog([(1, 0, 1), (1, 0, 3), (1, 2, 3)]))
    assert out["rating"].tolist() == [1, 3]
    assert out["day_offset"].tolist() == [0, 2]


def test_collapse_sorts_by_day_before_deduplicating():
    out = collapse_same_day(revlog([(1, 2, 3), (1, 0, 1), (1, 0, 4)]))
    assert out["rating"].tolist() == [1, 3]
```

- [ ] **Step 5: Rodar e ver falhar**

Run: `uv run pytest tests/test_series.py -v`
Expected: FAIL com `ImportError: cannot import name 'collapse_same_day'`.

- [ ] **Step 6: Implementar**

`src/fs/series.py`:

```python
"""Etapa 0: revlog de um usuário → uma linha por card elegível (janela + rótulo)."""
import numpy as np
import pandas as pd

N = 6      # revisões na janela de observação
H = 180    # dias do horizonte do rótulo
L = 2      # lapsos no horizonte para "card problemático"
CAP = 200  # teto de cards elegíveis por usuário


def collapse_same_day(revlog: pd.DataFrame) -> pd.DataFrame:
    """Uma revisão por (card, dia): fica a primeira em ordem cronológica.

    O arquivo já vem em ordem cronológica; o sort estável só protege contra entrada fora de ordem
    sem mudar a ordem dentro do mesmo dia.
    """
    ordered = revlog.sort_values("day_offset", kind="stable")
    return ordered.drop_duplicates(["card_id", "day_offset"], keep="first")
```

- [ ] **Step 7: Rodar e ver passar**

Run: `uv run pytest tests/test_series.py -v`
Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore src/fs
git commit -m "feat(series): projeto uv + colapso de revisões no mesmo dia"
```

---

### Task 2: Montar cards (janela, intervalos, rótulo, elegibilidade)

**Files:**
- Modify: `src/fs/series.py`
- Test: `tests/test_series.py`

**Interfaces:**
- Consumes: `collapse_same_day`, `N`, `H`, `L` (Task 1).
- Produces: `fs.series.COLUMNS: list[str]` =
  `["user_id", "card_id", "r0".."r5", "i0".."i5", "lapses_h", "n_reviews", "label"]`;
  `build_cards(revlog: pd.DataFrame, user_id: int, n: int = N, h: int = H, l: int = L) -> pd.DataFrame`
  com exatamente essas colunas, uma linha por card elegível. `r*` = nota (int), `i*` =
  `log1p(dias desde a revisão anterior)` (float, 0 na 1ª), `lapses_h` = nº de notas 1 no horizonte,
  `n_reviews` = revisões do card após colapsar, `label` = `lapses_h >= l`.

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar ao import de `tests/test_series.py`: `from fs.series import COLUMNS, build_cards, collapse_same_day`.
Acrescentar no fim do arquivo:

```python
# card 2 com uma revisão tardia só mantém o usuário ativo; ele próprio não é elegível (< 6 revisões)


def test_window_sees_only_first_six_reviews():
    rows = card(1, SIX_DAYS + [40], [3, 3, 3, 3, 3, 3, 1]) + card(2, [300], [3])
    row = build_cards(revlog(rows), 7).set_index("card_id").loc[1]
    assert [row[f"r{k}"] for k in range(6)] == [3] * 6
    assert row["lapses_h"] == 1
    assert not row["label"]  # 1 lapso < L = 2


def test_intervals_are_log1p_day_gaps_and_first_is_zero():
    rows = card(1, SIX_DAYS, [3] * 6) + card(2, [300], [3])
    row = build_cards(revlog(rows), 7).iloc[0]
    assert np.allclose([row[f"i{k}"] for k in range(6)], np.log1p([0, 1, 2, 4, 8, 15]))


def test_same_day_reviews_do_not_count_as_window_steps():
    rows = card(1, [0, 0, 1, 3, 7, 15, 30], [1, 3, 3, 3, 3, 3, 3]) + card(2, [300], [3])
    row = build_cards(revlog(rows), 7).iloc[0]
    assert [row[f"r{k}"] for k in range(6)] == [1, 3, 3, 3, 3, 3]


def test_label_horizon_is_open_at_start_and_closed_at_end():
    # 6ª revisão no dia 30: lapsos nos dias 100 e 210 contam; no dia 211, não
    rows = card(1, SIX_DAYS + [100, 210, 211], [3] * 6 + [1, 1, 1]) + card(2, [400], [3])
    row = build_cards(revlog(rows), 7).iloc[0]
    assert row["lapses_h"] == 2
    assert bool(row["label"]) is True


def test_lapses_inside_window_do_not_count_for_label():
    rows = card(1, SIX_DAYS + [100], [1, 1, 1, 1, 1, 1, 3]) + card(2, [300], [3])
    row = build_cards(revlog(rows), 7).iloc[0]
    assert row["lapses_h"] == 0
    assert not row["label"]


def test_card_with_fewer_than_six_reviews_is_excluded():
    rows = card(1, SIX_DAYS[:5], [3] * 5) + card(2, [300], [3])
    out = build_cards(revlog(rows), 7)
    assert out.empty
    assert list(out.columns) == COLUMNS


def test_card_excluded_when_user_stops_before_horizon_ends():
    rows = card(1, SIX_DAYS, [3] * 6) + card(2, [209], [3])  # último dia do usuário: 209 < 210
    assert build_cards(revlog(rows), 7).empty


def test_user_activity_on_other_cards_keeps_card_eligible():
    rows = card(1, SIX_DAYS, [3] * 6) + card(2, [210], [3])
    out = build_cards(revlog(rows), 7)
    assert out["card_id"].tolist() == [1]
    assert out["user_id"].tolist() == [7]
    assert out["lapses_h"].tolist() == [0]
    assert list(out.columns) == COLUMNS
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_series.py -v`
Expected: FAIL com `ImportError: cannot import name 'COLUMNS'`.

- [ ] **Step 3: Implementar**

Acrescentar em `src/fs/series.py`, depois de `collapse_same_day`:

```python
COLUMNS = [
    "user_id", "card_id",
    *[f"r{k}" for k in range(N)],
    *[f"i{k}" for k in range(N)],
    "lapses_h", "n_reviews", "label",
]


def build_cards(revlog: pd.DataFrame, user_id: int, n: int = N, h: int = H, l: int = L) -> pd.DataFrame:
    """Uma linha por card elegível: janela das n primeiras revisões + rótulo no horizonte de h dias."""
    last_day = revlog["day_offset"].max()  # censura pela atividade do USUÁRIO, não do card
    df = collapse_same_day(revlog).copy()
    df["k"] = df.groupby("card_id").cumcount()
    # intervalo entre dias distintos; elapsed_days do arquivo contaria as revisões colapsadas
    df["interval"] = df.groupby("card_id")["day_offset"].diff().fillna(0)

    t_n = df.loc[df["k"] == n - 1].set_index("card_id")["day_offset"].rename("t_n")
    t_n = t_n[t_n + h <= last_day]
    if t_n.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df.join(t_n, on="card_id", how="inner")

    window = df[df["k"] < n]
    ratings = window.pivot(index="card_id", columns="k", values="rating").add_prefix("r")
    intervals = np.log1p(window.pivot(index="card_id", columns="k", values="interval")).add_prefix("i")
    in_horizon = (df["day_offset"] > df["t_n"]) & (df["day_offset"] <= df["t_n"] + h)
    lapses = ((df["rating"] == 1) & in_horizon).groupby(df["card_id"]).sum().rename("lapses_h")
    n_reviews = df.groupby("card_id").size().rename("n_reviews")

    cards = pd.concat([ratings, intervals, lapses, n_reviews], axis=1).rename_axis(columns=None).reset_index()
    cards["user_id"] = user_id
    cards["label"] = cards["lapses_h"] >= l
    return cards[COLUMNS]
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_series.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fs/series.py
git commit -m "feat(series): cards com janela de 6 revisões e rótulo em horizonte de 180 dias"
```

---

### Task 3: Teto por usuário, sorteio/split de usuários e validação de schema

**Files:**
- Modify: `src/fs/series.py`
- Create: `src/fs/data.py`
- Test: `tests/test_series.py`

**Interfaces:**
- Consumes: `CAP` (Task 1).
- Produces:
  - `fs.series.cap_per_user(cards: pd.DataFrame, cap: int = CAP, seed: int = 0) -> pd.DataFrame`
    (ordenado por `user_id, card_id`, índice resetado);
  - `fs.data.REPO`, `REVISION`, `EXPECTED_COLUMNS: set[str]`, `SEED = 42`;
  - `fs.data.check_schema(df: pd.DataFrame) -> None` (levanta `ValueError` nomeando colunas faltando);
  - `fs.data.sample_users(n_users: int, seed: int = SEED) -> list[int]` (ordenada, ids em 1..10000);
  - `fs.data.split_users(users: list[int], seed: int = SEED) -> dict[int, str]`
    (valores `"train" | "val" | "test"`, 60/20/20).

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar aos imports de `tests/test_series.py`:

```python
from fs.data import check_schema, sample_users, split_users
from fs.series import COLUMNS, build_cards, cap_per_user, collapse_same_day
```

(substituindo a linha de import de `fs.series` anterior). Acrescentar no fim:

```python
def test_cap_limits_cards_per_user_and_is_deterministic():
    cards = pd.DataFrame({"user_id": [1] * 10 + [2] * 3, "card_id": range(13)})
    a = cap_per_user(cards, cap=4, seed=0)
    assert a.groupby("user_id").size().to_dict() == {1: 4, 2: 3}
    pd.testing.assert_frame_equal(a, cap_per_user(cards, cap=4, seed=0))


def test_split_users_is_complete_proportional_and_deterministic():
    users = list(range(1, 101))
    s = split_users(users)
    assert set(s) == set(users)  # dict: cada usuário em exatamente um split
    assert list(s.values()).count("train") == 60
    assert list(s.values()).count("val") == 20
    assert list(s.values()).count("test") == 20
    assert s == split_users(users)


def test_sample_users_is_deterministic_and_in_range():
    a = sample_users(50)
    assert a == sample_users(50)
    assert len(set(a)) == 50 and min(a) >= 1 and max(a) <= 10000


def test_check_schema_names_missing_columns():
    with pytest.raises(ValueError, match="rating"):
        check_schema(revlog([(1, 0, 3)]).drop(columns="rating"))
    check_schema(revlog([(1, 0, 3)]))  # schema completo não levanta
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_series.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'fs.data'`.

- [ ] **Step 3: Implementar `cap_per_user`**

Acrescentar no fim de `src/fs/series.py`:

```python
def cap_per_user(cards: pd.DataFrame, cap: int = CAP, seed: int = 0) -> pd.DataFrame:
    """No máximo `cap` cards por usuário, sorteados — nenhum usuário domina treino ou métrica."""
    shuffled = cards.sample(frac=1, random_state=seed)
    capped = shuffled.groupby("user_id").head(cap)
    return capped.sort_values(["user_id", "card_id"]).reset_index(drop=True)
```

- [ ] **Step 4: Criar `src/fs/data.py`**

```python
"""Etapa 0: sorteia usuários, baixa os revlogs, monta os cards e escreve reports/etapa0.md."""
import random

import pandas as pd

REPO = "open-spaced-repetition/anki-revlogs-10k"
REVISION = "75299740cff05894ef42d7ad990666691efdd2da"
EXPECTED_COLUMNS = {"card_id", "day_offset", "rating", "state", "duration", "elapsed_days", "elapsed_seconds"}
SEED = 42


def check_schema(df: pd.DataFrame) -> None:
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"schema inesperado no revlog: faltam {sorted(missing)}. "
            f"O dataset mudou? Confira REVISION ({REVISION})."
        )


def sample_users(n_users: int, seed: int = SEED) -> list[int]:
    return sorted(random.Random(seed).sample(range(1, 10001), n_users))


def split_users(users: list[int], seed: int = SEED) -> dict[int, str]:
    """60/20/20 por usuário: cada pessoa dá nota do seu jeito, então ninguém cruza de split."""
    shuffled = list(users)
    random.Random(seed).shuffle(shuffled)
    a, b = int(len(shuffled) * 0.6), int(len(shuffled) * 0.8)
    return {u: "train" if i < a else "val" if i < b else "test" for i, u in enumerate(shuffled)}
```

- [ ] **Step 5: Rodar e ver passar**

Run: `uv run pytest tests/test_series.py -v`
Expected: 14 passed.

- [ ] **Step 6: Commit**

```bash
git add src/fs/series.py src/fs/data.py 
git commit -m "feat(data): teto por usuário, split 60/20/20 por usuário e validação de schema"
```

---

### Task 4: Download, pipeline e relatório da Etapa 0

**Files:**
- Modify: `src/fs/data.py`
- Create: `README.md` (substitui o atual, se houver), `reports/etapa0.md`, `reports/etapa0/*.png` (gerados)

**Interfaces:**
- Consumes: `build_cards`, `cap_per_user`, `N`, `H`, `L`, `CAP` (series); `check_schema`,
  `sample_users`, `split_users`, `REPO`, `REVISION`, `SEED` (data).
- Produces: `data/cards.parquet` com `COLUMNS + ["split"]` (fora do git) — é a entrada das Etapas 1 e 2;
  `fs.data.load_revlog(user_id: int) -> pd.DataFrame`; CLI `uv run python -m fs.data [--n-users N]`.

- [ ] **Step 1: Implementar download, pipeline e relatório**

Substituir o bloco de imports de `src/fs/data.py` e acrescentar o restante no fim:

```python
"""Etapa 0: sorteia usuários, baixa os revlogs, monta os cards e escreve reports/etapa0.md."""
import argparse
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

from fs.series import CAP, H, L, N, build_cards, cap_per_user
```

```python
DATA = Path("data")
REPORTS = Path("reports")


def load_revlog(user_id: int) -> pd.DataFrame:
    try:
        path = hf_hub_download(
            REPO, f"revlogs/user_id={user_id}/data.parquet", repo_type="dataset", revision=REVISION
        )
    except GatedRepoError as e:
        raise SystemExit(
            "Sem acesso ao dataset (gated). Aceite os termos em "
            f"https://huggingface.co/datasets/{REPO} e salve um token de leitura em "
            "~/.cache/huggingface/token."
        ) from e
    df = pd.read_parquet(path)
    check_schema(df)
    return df


def write_report(eligible: pd.DataFrame, cards: pd.DataFrame, users: list[int], total_reviews: int, seconds: float) -> None:
    figs = REPORTS / "etapa0"
    figs.mkdir(parents=True, exist_ok=True)
    train = cards[cards["split"] == "train"]
    per_user = eligible.groupby("user_id").size().reindex(users, fill_value=0)
    top2 = per_user.nlargest(2).sum() / max(len(eligible), 1)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(cards["n_reviews"], bins=50, log=True)
    ax.set(xlabel="revisões por card (após colapsar o mesmo dia)", ylabel="cards (escala log)")
    fig.tight_layout()
    fig.savefig(figs / "n_reviews.png", dpi=150)
    plt.close(fig)

    # só figura: a licença proíbe redistribuir o dado, então nada de tabela com séries brutas
    normal, prob = train[~train["label"]], train[train["label"]]
    examples = pd.concat([normal.sample(min(2, len(normal)), random_state=0),
                          prob.sample(min(1, len(prob)), random_state=0)])
    fig, axes = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
    for _, row in examples.iterrows():
        tag = "problemático" if row["label"] else "normal"
        axes[0].plot(range(N), [row[f"r{k}"] for k in range(N)], marker="o", label=tag)
        axes[1].plot(range(N), [row[f"i{k}"] for k in range(N)], marker="o")
    axes[0].set(ylabel="nota (1 = esqueceu)", yticks=[1, 2, 3, 4])
    axes[0].legend()
    axes[1].set(ylabel="log(1 + dias)", xlabel="revisão da janela")
    fig.tight_layout()
    fig.savefig(figs / "series_exemplo.png", dpi=150)
    plt.close(fig)

    splits = cards.groupby("split")["user_id"].nunique().to_dict()
    prev = {l: (train["lapses_h"] >= l).mean() for l in (1, 2, 3)}
    lines = [
        "# Etapa 0 — Base comum",
        "",
        f"Gerado por `uv run python -m fs.data --n-users {len(users)}` em {seconds / 60:.1f} min.",
        f"Dataset `{REPO}` @ `{REVISION[:12]}` · seed {SEED} · N = {N}, H = {H} dias, L = {L}, teto = {CAP}.",
        "",
        "| | |",
        "|---|---|",
        f"| Usuários sorteados | {len(users)} |",
        f"| Usuários por split (com algum card) | treino {splits.get('train', 0)} · val {splits.get('val', 0)} · teste {splits.get('test', 0)} |",
        f"| Usuários sem nenhum card elegível | {(per_user == 0).sum()} |",
        f"| Revisões brutas baixadas | {total_reviews:,} |",
        f"| Cards elegíveis antes do teto | {len(eligible):,} (top-2 usuários: {top2:.0%}) |",
        f"| Cards elegíveis por usuário | mediana {per_user.median():.0f} · p90 {per_user.quantile(0.9):.0f} · máx {per_user.max()} |",
        f"| Cards após o teto | {len(cards):,} |",
        f"| Prevalência de card problemático no **treino** (L = {L}) | **{prev[L]:.1%}** |",
        f"| Robustez no treino: L = 1 · L = 3 | {prev[1]:.1%} · {prev[3]:.1%} |",
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
        print(f"[{i}/{len(users)}] user {u}: {len(parts[-1])} cards elegíveis", flush=True)

    eligible = pd.concat([p for p in parts if len(p)], ignore_index=True)
    cards = cap_per_user(eligible)
    cards["split"] = cards["user_id"].map(split)
    DATA.mkdir(exist_ok=True)
    cards.to_parquet(DATA / "cards.parquet", index=False)
    write_report(eligible, cards, users, total_reviews, time.time() - start)
    print(f"ok: {len(cards)} cards em {DATA / 'cards.parquet'} · relatório em {REPORTS / 'etapa0.md'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rodar os testes (nada pode ter quebrado)**

Run: `uv run pytest -v`
Expected: 14 passed.

- [ ] **Step 3: Smoke test com 5 usuários**

Run: `uv run python -m fs.data --n-users 5`
Expected: 5 linhas `[i/5] user …`, depois `ok: … cards`. Arquivos `data/cards.parquet`,
`reports/etapa0.md`, `reports/etapa0/n_reviews.png`, `reports/etapa0/series_exemplo.png` existem.
Abrir as duas figuras e conferir que não estão vazias.

- [ ] **Step 4: Conferir que o dado não entra no git**

Run: `git status --short`
Expected: aparecem `reports/` e `src/fs/data.py`; **não** aparece nada de `data/`.

- [ ] **Step 5: Conferir a mensagem de acesso negado**

Run: `HF_HOME="$(mktemp -d)" uv run python -m fs.data --n-users 1`
Expected: termina com a mensagem "Sem acesso ao dataset (gated)…", **sem traceback de 401**.
Se aparecer outro erro (ex.: `HfHubHTTPError` 401 em vez de `GatedRepoError`), ampliar o `except`
para essa classe e rodar de novo.

- [ ] **Step 6: Rodada completa, cronometrada**

Run: `time uv run python -m fs.data`
Expected: termina em **< 30 min**. Se passar disso, **parar e reportar** (a spec manda reduzir
usuários, e isso é decisão da Ana).

- [ ] **Step 7: Conferir os números contra a sonda**

Abrir `reports/etapa0.md`. Referência da sonda de 30 usuários (2026-10-02): prevalência com L = 2 de
**~7–9%**, L = 1 de ~18%, L = 3 de 4–5%, quase todo usuário com algum card elegível.
Se a prevalência de L = 2 no treino ficar **fora de 4–15%**, **parar e reportar** em vez de mexer em
N, H ou L — mudar o rótulo depois de ver o dado é decisão de spec.

- [ ] **Step 8: README mínimo**

`README.md`:

````markdown
# forgetting-shapelets

Dá para saber, pelas **6 primeiras revisões** de um flashcard, que ele vai ser esquecido de novo e
de novo? Este projeto testa **shapelets** (padrões curtos em séries temporais) contra baselines
honestos, sobre logs reais do Anki.

> **Status:** Etapa 0 (base de dados) concluída — ver [`reports/etapa0.md`](reports/etapa0.md).
> Etapa 1 (classificação de cards problemáticos) em andamento.

## Dado

[`open-spaced-repetition/anki-revlogs-10k`](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k)
— logs de revisão de 10 mil usuários do Anki. A licença permite uso em pesquisa por estudantes e
indivíduos e **proíbe redistribuição**: o dado não está neste repo, só os relatórios agregados.

## Como reproduzir

1. Aceite os termos do dataset no Hugging Face e salve um token de leitura em `~/.cache/huggingface/token`.
2. `uv sync`
3. `uv run python -m fs.data` (Etapa 0; ~1 GB de download)

Testes: `uv run pytest`.

Desenho completo: [`docs/superpowers/specs/2026-10-01-forgetting-shapelets-design.md`](docs/superpowers/specs/2026-10-01-forgetting-shapelets-design.md)
· versão simples: [`…-explicado.md`](docs/superpowers/specs/2026-10-01-forgetting-shapelets-explicado.md).
````

- [ ] **Step 9: Commit**

```bash
git add src/fs/data.py reports/
git status --short   # conferir de novo: nada de data/
git commit -m "feat(data): pipeline básico e relatório da base (500 usuários)"
```

---

## Fora deste plano

- Checkpoint 3 da spec (RDST com séries de tamanho desigual) — pertence à Etapa 2.
- aeon, fsrs e scikit-learn — entram no plano da Etapa 1.
- `data/cards.parquet` só tem a janela de 6 revisões. A Etapa 2 vai precisar do histórico completo;
  o plano dela decide se estende `build_cards` ou lê os revlogs de novo (estão no cache do HF).
