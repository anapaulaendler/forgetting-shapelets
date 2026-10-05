# Etapa 1 — Card problemático · Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Como este plano é executado:** a **Ana escreve o código**; o Claude entrega uma colinha por task
> (`colinha/task-N/`, fora do git) e revisa o código dela. **Uma task por vez**, com aprovação antes da
> próxima (a menos que a mensagem traga `--skip-validations`).

**Goal:** responder, com IC honesto, se shapelets sobre as 6 primeiras revisões de um card preveem que
ele será "problemático" (≥ 2 lapsos nos 180 dias seguintes) melhor que baselines — e mostrar os 5
padrões que o modelo aprendeu.

**Architecture:** `src/fs/leech.py` monta três conjuntos de features a partir de `data/cards.parquet`
(agregados da janela, estado FSRS, série 2×6), treina a escada B0→B1→B2→M nos usuários de treino
escolhendo `C` na validação, e avalia no teste. `src/fs/evaluate.py` (métricas + bootstrap por usuário)
fica separado porque a Etapa 2 vai reusar.

**Tech Stack:** Python 3.12 · pandas · scikit-learn ≥ 1.5 · aeon ≥ 1.6 (`RandomDilatedShapeletTransform`)
· fsrs ≥ 6.3 (py-fsrs) · matplotlib · pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-forgetting-shapelets-design.md` (§4 Etapa 1, §7, §8)
**Depende de:** Etapa 0 concluída (`docs/superpowers/plans/2026-10-02-etapa0-base-comum.md`).

## Global Constraints

- Entrada: `data/cards.parquet` da Etapa 0 — colunas `user_id, card_id, r0..r5, i0..i5, lapses_h,
  n_reviews, label, split`. `r*` = nota 1–4; `i*` = `log1p(dias desde a anterior)`, `i0 = 0`.
- **N = 6, H = 180, L = 2**; robustez com **L = 1 e L = 3** (rótulo = `lapses_h >= L`).
- Split por usuário já feito na Etapa 0 (60/20/20). Treino ajusta, **validação escolhe `C`**, **teste só
  avalia** — nunca escolhe nada.
- Escada: **B0** prevalência do treino · **B1** logística sobre agregados da janela · **B2** B1 + estado
  FSRS (stability, difficulty) com **parâmetros default** · **M** RDST → logística.
- Logística (não ridge) depois do RDST, porque as métricas exigem probabilidade.
- Métrica principal **PR-AUC**; secundárias ROC-AUC, Brier, curva de calibração.
- **IC 95% por bootstrap reamostrando usuários do teste**, 1000 amostras, as mesmas amostras para todos
  os modelos.
- **M vence B2 só se o IC 95% da diferença de PR-AUC excluir zero**; senão o README declara empate/perda.
- Top-5 shapelets traduzidos em padrão legível — **ganhando ou não**.
- Licença: `reports/` só com agregados, figuras e parâmetros de modelo; nada de linhas do dataset.
- Etapa roda em **< 30 min**.

**Decisões deste plano** (a spec não fixa; medidas em sonda de 2026-10-05):
- RDST com `shapelet_lengths=[2, 3, 4]` (o default do aeon é 11 e não cabe em série de 6),
  `max_shapelets=200`, **`proba_normalization=0.0`** (shapelets em nota/dias de verdade, legíveis — o
  default normaliza 80% deles), `random_state=0`. Fit em 50 mil séries: ~1 s.
- `C` da logística ∈ {0.01, 0.1, 1, 10}, com `StandardScaler` antes.
- FSRS reconstruído a partir de `i*` (`dias = round(expm1(i))`), `Scheduler(enable_fuzzing=False)`,
  **`Card(card_id=k)` explícito** — sem id, o py-fsrs dorme 1 ms por card para gerar um id único (80 mil
  cards: 1,5 min → 3 s).

## Review Focus

1. **Reamostra do bootstrap sem nenhum positivo** (usuários sorteados sem card problemático) → a métrica
   vira `NaN` e o IC ignora essa amostra, em vez de quebrar. *Teste na Task 3.*
2. **Intervalos longos na reconstrução do FSRS** (ex.: 3000 dias) → `expm1`/`round` devolve o número
   exato de dias. *Teste na Task 2.*
3. **`cards.parquet` da Etapa 0 com nomes diferentes do plano** → aborta dizendo quais colunas faltam.
   *Teste na Task 1.*
4. **Shapelet que só se distingue pela ordem** (dois "esqueci" seguidos vs. separados) → M encontra, B1
   não. É o teste de que o RDST está ligado direito. *Teste na Task 4.*
5. **Mapeamento coluna → shapelet** (cada shapelet gera 3 colunas) → o top-5 aponta o shapelet certo.
   *Teste na Task 5.*

---

### Task 1: Dependências + features da janela (B1) + série (M)

**Files:**
- Modify: `pyproject.toml`
- Create: `src/fs/leech.py`
- Test: `tests/test_leech.py`

**Interfaces:**
- Consumes: `fs.series.N`, `fs.series.COLUMNS` (Etapa 0).
- Produces: `fs.leech.RATING_COLS`, `INTERVAL_COLS` (`list[str]`);
  `check_cards(cards: pd.DataFrame) -> None`;
  `to_series(cards: pd.DataFrame) -> np.ndarray` com shape `(n, 2, N)`;
  `window_features(cards: pd.DataFrame) -> pd.DataFrame` com colunas
  `n_lapsos, nota_media, ultima_nota, ultimo_intervalo, crescimento_intervalo` (mesmo índice de `cards`).

- [ ] **Step 0: Conferir a Etapa 0**

Se os nomes no seu `fs/series.py` diferem do plano da Etapa 0 (ex.: `HORIZON_DAYS` em vez de `H`), use os
seus no lugar dos deste plano. Confira as colunas reais:

Run: `uv run python -c "import pandas as pd; print(pd.read_parquet('data/cards.parquet').columns.tolist())"`
Expected: `['user_id', 'card_id', 'r0', …, 'r5', 'i0', …, 'i5', 'lapses_h', 'n_reviews', 'label', 'split']`.

- [ ] **Step 1: Adicionar dependências**

Em `pyproject.toml`, na lista `dependencies`, acrescentar:

```toml
    "scikit-learn>=1.5",
    "aeon>=1.6",
    "fsrs>=6.3",
```

Run: `uv sync && uv run python -c "import aeon, fsrs, sklearn; print(aeon.__version__)"`
Expected: imprime `1.6.0` ou maior.

- [ ] **Step 2: Escrever os testes que falham**

`tests/test_leech.py`:

```python
import numpy as np
import pandas as pd
import pytest

from fs.leech import INTERVAL_COLS, RATING_COLS, check_cards, to_series, window_features


def make_cards(ratings, gaps, users=None, splits=None):
    """ratings/gaps: uma lista de 6 por card; gaps em dias desde a revisão anterior (o 1º é 0)."""
    n = len(ratings)
    df = pd.DataFrame(np.asarray(ratings), columns=RATING_COLS)
    df[INTERVAL_COLS] = np.log1p(np.asarray(gaps, dtype=float))
    df["user_id"] = np.arange(n) if users is None else users
    df["card_id"] = np.arange(n)
    df["lapses_h"] = 0
    df["n_reviews"] = 6
    df["label"] = False
    df["split"] = "train" if splits is None else splits
    return df


GAPS = [0, 1, 2, 4, 8, 15]


def test_to_series_stacks_ratings_and_intervals_as_two_channels():
    cards = make_cards([[3, 1, 3, 1, 3, 4], [2, 2, 2, 2, 2, 2]], [GAPS, GAPS])
    x = to_series(cards)
    assert x.shape == (2, 2, 6)
    assert x[0, 0].tolist() == [3, 1, 3, 1, 3, 4]
    assert np.allclose(x[0, 1], np.log1p(GAPS))


def test_window_features():
    f = window_features(make_cards([[3, 1, 3, 1, 3, 4]], [GAPS])).iloc[0]
    assert f["n_lapsos"] == 2
    assert f["nota_media"] == pytest.approx(2.5)
    assert f["ultima_nota"] == 4
    assert f["ultimo_intervalo"] == pytest.approx(np.log1p(15))
    assert f["crescimento_intervalo"] == pytest.approx(np.log1p(15) - np.log1p(1))


def test_check_cards_names_missing_columns():
    cards = make_cards([[3] * 6], [GAPS])
    check_cards(cards)  # completo: não levanta
    with pytest.raises(ValueError, match="lapses_h"):
        check_cards(cards.drop(columns="lapses_h"))
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_leech.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'fs.leech'`.

- [ ] **Step 4: Implementar**

`src/fs/leech.py`:

```python
"""Etapa 1: as 6 primeiras revisões preveem um card problemático (≥ L lapsos em 180 dias)?"""
import numpy as np
import pandas as pd

from fs.series import COLUMNS, N

RATING_COLS = [f"r{k}" for k in range(N)]
INTERVAL_COLS = [f"i{k}" for k in range(N)]


def check_cards(cards: pd.DataFrame) -> None:
    missing = set(COLUMNS + ["split"]) - set(cards.columns)
    if missing:
        raise ValueError(
            f"data/cards.parquet sem as colunas {sorted(missing)}. "
            "Rode a Etapa 0 de novo: uv run python -m fs.data"
        )


def to_series(cards: pd.DataFrame) -> np.ndarray:
    """(n_cards, 2 canais, N passos): canal 0 = nota, canal 1 = log(1 + dias desde a anterior)."""
    return np.stack(
        [cards[RATING_COLS].to_numpy(float), cards[INTERVAL_COLS].to_numpy(float)], axis=1
    )


def window_features(cards: pd.DataFrame) -> pd.DataFrame:
    """B1: agregados da janela — o que dá para ver sem olhar a ORDEM das revisões."""
    r = cards[RATING_COLS].to_numpy()
    i = cards[INTERVAL_COLS].to_numpy()
    return pd.DataFrame(
        {
            "n_lapsos": (r == 1).sum(axis=1),
            "nota_media": r.mean(axis=1),
            "ultima_nota": r[:, -1],
            "ultimo_intervalo": i[:, -1],
            "crescimento_intervalo": i[:, -1] - i[:, 1],  # i0 é sempre 0
        },
        index=cards.index,
    )
```

- [ ] **Step 5: Rodar e ver passar**

Run: `uv run pytest tests/test_leech.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/fs/leech.py tests/test_leech.py
git commit -m "feat(leech): dependências, features da janela (B1) e série 2×6 (M)"
```

---

### Task 2: Estado FSRS ao fim da janela (B2)

**Files:**
- Modify: `src/fs/leech.py`
- Test: `tests/test_leech.py`

**Interfaces:**
- Consumes: `INTERVAL_COLS`, `RATING_COLS` (Task 1).
- Produces: `review_days(cards) -> np.ndarray` `(n, N)` int, dia de cada revisão contado da 1ª;
  `fsrs_state(cards) -> pd.DataFrame` com colunas `stability, difficulty` (mesmo índice de `cards`).

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar ao import: `from fs.leech import … , fsrs_state, review_days`. No topo do arquivo:
`from datetime import datetime, timedelta, timezone` e `from fsrs import Card, Rating, Scheduler`.
Acrescentar no fim:

```python
def test_review_days_round_trips_even_long_gaps():
    cards = make_cards([[3] * 6], [[0, 1, 2, 4, 8, 3000]])
    assert review_days(cards)[0].tolist() == [0, 1, 3, 7, 15, 3015]


def test_fsrs_state_matches_direct_replay():
    ratings, gaps = [3, 1, 3, 3, 3, 3], [0, 1, 2, 4, 8, 15]
    state = fsrs_state(make_cards([ratings], [gaps])).iloc[0]

    scheduler, card = Scheduler(enable_fuzzing=False), Card(card_id=1)
    base = datetime(2000, 1, 1, tzinfo=timezone.utc)
    for day, r in zip(np.cumsum(gaps), ratings):
        card, _ = scheduler.review_card(card, Rating(r), review_datetime=base + timedelta(days=int(day)))
    assert state["stability"] == pytest.approx(card.stability)
    assert state["difficulty"] == pytest.approx(card.difficulty)


def test_fsrs_state_forgotten_card_is_less_stable():
    s = fsrs_state(make_cards([[1] * 6, [3] * 6], [GAPS, GAPS]))
    assert s["stability"].iloc[0] < s["stability"].iloc[1]
    assert s["difficulty"].iloc[0] > s["difficulty"].iloc[1]
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_leech.py -v`
Expected: FAIL com `ImportError: cannot import name 'fsrs_state'`.

- [ ] **Step 3: Implementar**

Em `src/fs/leech.py`, acrescentar aos imports:

```python
from datetime import datetime, timedelta, timezone

from fsrs import Card, Rating, Scheduler
```

E no fim do arquivo:

```python
BASE_DATE = datetime(2000, 1, 1, tzinfo=timezone.utc)  # só a diferença entre dias importa


def review_days(cards: pd.DataFrame) -> np.ndarray:
    """Dia de cada revisão (contado da 1ª), desfazendo o log1p dos intervalos."""
    gaps = np.rint(np.expm1(cards[INTERVAL_COLS].to_numpy(float))).astype(int)
    return gaps.cumsum(axis=1)


def fsrs_state(cards: pd.DataFrame) -> pd.DataFrame:
    """B2: estabilidade e dificuldade do FSRS depois de reproduzir as N revisões da janela.

    Parâmetros default (a spec não otimiza o FSRS na Etapa 1). card_id explícito: sem ele o py-fsrs
    dorme 1 ms por card para gerar um id único.
    """
    scheduler = Scheduler(enable_fuzzing=False)
    rows = []
    for k, (days, ratings) in enumerate(zip(review_days(cards), cards[RATING_COLS].to_numpy(int))):
        card = Card(card_id=k)
        for day, r in zip(days, ratings):
            card, _ = scheduler.review_card(
                card, Rating(int(r)), review_datetime=BASE_DATE + timedelta(days=int(day))
            )
        rows.append((card.stability, card.difficulty))
    return pd.DataFrame(rows, columns=["stability", "difficulty"], index=cards.index)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_leech.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fs/leech.py tests/test_leech.py
git commit -m "feat(leech): estado FSRS ao fim da janela (B2)"
```

---

### Task 3: Métricas e IC por bootstrap de usuários

**Files:**
- Create: `src/fs/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces (reusado pela Etapa 2):
  - `metrics(y, p) -> dict[str, float]` com chaves `pr_auc, roc_auc, brier` (`NaN` quando indefinida);
  - `user_groups(users) -> list[np.ndarray]` (índices de cada usuário);
  - `resample_users(groups, rng) -> np.ndarray` (índices de uma amostra bootstrap);
  - `bootstrap(users, y, preds: dict[str, np.ndarray], n_boot=1000, seed=0) -> pd.DataFrame`
    com colunas `boot, model, metric, value`;
  - `ci_table(y, preds, samples) -> pd.DataFrame` com colunas `model, metric, point, lo, hi`;
  - `diff_ci(samples, a: str, b: str, metric="pr_auc") -> tuple[float, float]`;
  - `verdict(lo: float, hi: float) -> str` ∈ `{"vence", "perde", "empate"}`.

- [ ] **Step 1: Escrever os testes que falham**

`tests/test_evaluate.py`:

```python
import numpy as np

from fs.evaluate import bootstrap, ci_table, diff_ci, metrics, resample_users, user_groups, verdict


def test_metrics_perfect_predictor():
    m = metrics([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0])
    assert m == {"pr_auc": 1.0, "roc_auc": 1.0, "brier": 0.0}


def test_metrics_without_positives_is_nan_not_crash():
    m = metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert np.isnan(m["pr_auc"]) and np.isnan(m["roc_auc"])


def test_resample_takes_whole_users():
    users = np.array([1, 1, 1, 2, 3, 3])
    sizes = {1: 3, 2: 1, 3: 2}
    groups, rng = user_groups(users), np.random.default_rng(0)
    for _ in range(50):
        drawn = users[resample_users(groups, rng)]
        blocks = 0
        for u, size in sizes.items():
            assert (drawn == u).sum() % size == 0  # usuário entra com TODOS os cards, ou não entra
            blocks += (drawn == u).sum() // size
        assert blocks == 3  # sorteia tantos usuários quanto existem


def test_same_model_twice_has_zero_diff():
    rng = np.random.default_rng(1)
    users, y, p = np.repeat(np.arange(20), 10), rng.integers(0, 2, 200), rng.random(200)
    s = bootstrap(users, y, {"a": p, "b": p.copy()}, n_boot=50)
    assert diff_ci(s, "a", "b") == (0.0, 0.0)


def test_better_model_wins_and_ci_brackets_point():
    rng = np.random.default_rng(2)
    users, y = np.repeat(np.arange(40), 25), rng.integers(0, 2, 1000)
    preds = {"bom": y * 0.6 + rng.random(1000) * 0.4, "ruim": rng.random(1000)}
    s = bootstrap(users, y, preds, n_boot=200)
    assert verdict(*diff_ci(s, "bom", "ruim")) == "vence"
    t = ci_table(y, preds, s).set_index(["model", "metric"])
    for _, row in t.iterrows():
        assert row["lo"] <= row["point"] <= row["hi"]


def test_bootstrap_survives_samples_without_positives():
    users, y = np.array([1, 2, 3, 4]), np.array([1, 0, 0, 0])  # só o usuário 1 tem positivo
    s = bootstrap(users, y, {"a": np.array([0.9, 0.1, 0.2, 0.3])}, n_boot=100)
    pr = s[s.metric == "pr_auc"]["value"]
    assert pr.isna().any() and pr.notna().any()
    t = ci_table(y, {"a": np.array([0.9, 0.1, 0.2, 0.3])}, s)
    assert t["lo"].notna().all()


def test_verdict():
    assert verdict(0.01, 0.05) == "vence"
    assert verdict(-0.05, -0.01) == "perde"
    assert verdict(-0.01, 0.02) == "empate"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'fs.evaluate'`.

- [ ] **Step 3: Implementar**

`src/fs/evaluate.py`:

```python
"""Métricas e IC 95% por bootstrap de usuários — compartilhado pelas Etapas 1 e 2."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

N_BOOT = 1000


def metrics(y, p) -> dict[str, float]:
    """PR-AUC e ROC-AUC viram NaN quando a amostra não tem as duas classes."""
    y, p = np.asarray(y), np.asarray(p)
    both = 0 < y.sum() < len(y)
    return {
        "pr_auc": average_precision_score(y, p) if both else np.nan,
        "roc_auc": roc_auc_score(y, p) if both else np.nan,
        "brier": brier_score_loss(y, p),
    }


def user_groups(users) -> list[np.ndarray]:
    users = np.asarray(users)
    return list(pd.Series(users).groupby(users).indices.values())


def resample_users(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Sorteia usuários com reposição e leva TODOS os cards de cada um: cards da mesma pessoa são
    correlacionados, então a unidade de incerteza é a pessoa, não o card."""
    drawn = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[g] for g in drawn])


def bootstrap(users, y, preds: dict[str, np.ndarray], n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Uma linha por (amostra, modelo, métrica). Todos os modelos usam as MESMAS amostras — é isso
    que permite o IC da diferença entre dois modelos."""
    y, groups, rng = np.asarray(y), user_groups(users), np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        idx = resample_users(groups, rng)
        for name, p in preds.items():
            for metric, value in metrics(y[idx], np.asarray(p)[idx]).items():
                rows.append((b, name, metric, value))
    return pd.DataFrame(rows, columns=["boot", "model", "metric", "value"])


def ci_table(y, preds: dict[str, np.ndarray], samples: pd.DataFrame) -> pd.DataFrame:
    """Ponto (teste inteiro) + IC 95% percentil, ignorando amostras em que a métrica é indefinida."""
    rows = []
    for name, p in preds.items():
        for metric, point in metrics(y, p).items():
            v = samples.loc[(samples.model == name) & (samples.metric == metric), "value"]
            rows.append((name, metric, point, np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)))
    return pd.DataFrame(rows, columns=["model", "metric", "point", "lo", "hi"])


def diff_ci(samples: pd.DataFrame, a: str, b: str, metric: str = "pr_auc") -> tuple[float, float]:
    """IC 95% de (a − b). Para PR-AUC/ROC-AUC, positivo = a melhor; para Brier, o contrário."""
    wide = samples[samples.metric == metric].pivot(index="boot", columns="model", values="value")
    d = (wide[a] - wide[b]).to_numpy()
    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def verdict(lo: float, hi: float) -> str:
    """Regra da spec: só vence se o IC da diferença excluir zero."""
    if lo > 0:
        return "vence"
    if hi < 0:
        return "perde"
    return "empate"
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fs/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): métricas e IC 95% por bootstrap de usuários"
```

---

### Task 4: A escada de modelos (B0 → B1 → B2 → M)

**Files:**
- Modify: `src/fs/leech.py`
- Test: `tests/test_leech.py`

**Interfaces:**
- Consumes: `window_features`, `fsrs_state`, `to_series` (Tasks 1–2); `metrics` (Task 3).
- Produces:
  - constantes `MAX_SHAPELETS = 200`, `SHAPELET_LENGTHS = [2, 3, 4]`, `CS = (0.01, 0.1, 1.0, 10.0)`, `SEED = 0`;
  - `build_features(cards) -> dict[str, np.ndarray]` com chaves `"B1"` `(n, 5)`, `"B2"` `(n, 7)`, `"series"` `(n, 2, N)`;
  - `fit_logreg(X_train, y_train, X_val, y_val) -> sklearn.pipeline.Pipeline`;
  - `fit_rdst(series, y) -> RandomDilatedShapeletTransform`;
  - `run_ladder(features, split: np.ndarray, y: np.ndarray) -> tuple[dict[str, np.ndarray], dict]` —
    1º: `{"B0"|"B1"|"B2"|"M": P(problemático) nos cards de teste}`; 2º: `{"rdst": transform, "B1"|"B2"|"M": Pipeline}`.

- [ ] **Step 1: Escrever o teste que falha**

Acrescentar ao import: `from fs.evaluate import metrics` e `from fs.leech import … , build_features, run_ladder`.
Acrescentar no fim:

```python
def planted_cards(n_users=30, per_user=50, seed=0):
    """Positivo = dois 'esqueci' SEGUIDOS; negativo = dois 'esqueci' SEPARADOS.
    Mesmo nº de lapsos nos dois: só quem enxerga a ordem consegue separar."""
    rng = np.random.default_rng(seed)
    ratings, labels = [], []
    for k in range(n_users * per_user):
        r = rng.integers(2, 5, 6)
        positive = k % 2 == 0
        a = rng.integers(0, 5)
        b = a + 1 if positive else (a + 2 + rng.integers(0, 6 - a - 2)) if a <= 3 else None
        if b is None:  # a = 4 não tem posição separada depois; usa a anterior
            a, b = 1, 4
        r[[a, b]] = 1
        ratings.append(r)
        labels.append(positive)
    users = np.repeat(np.arange(n_users), per_user)
    splits = np.where(users < 18, "train", np.where(users < 24, "val", "test"))
    gaps = rng.integers(1, 30, (len(ratings), 6))
    gaps[:, 0] = 0
    return make_cards(ratings, gaps, users=users, splits=splits), np.array(labels)


def test_shapelets_see_order_that_aggregates_cannot():
    cards, y = planted_cards()
    preds, fitted = run_ladder(build_features(cards), cards["split"].to_numpy(), y)
    y_test = y[cards["split"].to_numpy() == "test"]
    pr = {name: metrics(y_test, p)["pr_auc"] for name, p in preds.items()}
    assert pr["M"] > 0.9
    assert pr["M"] > pr["B1"] + 0.1
    assert np.allclose(preds["B0"], y[cards["split"].to_numpy() == "train"].mean())
    assert set(fitted) == {"rdst", "B1", "B2", "M"}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_leech.py::test_shapelets_see_order_that_aggregates_cannot -v`
Expected: FAIL com `ImportError: cannot import name 'build_features'`.

- [ ] **Step 3: Implementar**

Em `src/fs/leech.py`, acrescentar aos imports:

```python
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from fs.evaluate import metrics
```

E no fim:

```python
# ponytail: 200 shapelets × 3 features já cobrem padrões de tamanho ≤ 4 em séries de 6;
# subir só se M ficar colado em B2
MAX_SHAPELETS = 200
SHAPELET_LENGTHS = [2, 3, 4]  # o default do aeon (11) não cabe numa série de 6
CS = (0.01, 0.1, 1.0, 10.0)
SEED = 0


def build_features(cards: pd.DataFrame) -> dict[str, np.ndarray]:
    """Calculado uma vez e reusado para L = 1, 2, 3 — as features não dependem do rótulo."""
    b1 = window_features(cards)
    return {
        "B1": b1.to_numpy(float),
        "B2": b1.join(fsrs_state(cards)).to_numpy(float),
        "series": to_series(cards),
    }


def fit_logreg(X_train, y_train, X_val, y_val) -> Pipeline:
    """Escolhe C pela PR-AUC na validação. O teste nunca participa da escolha."""
    best, best_score = None, -np.inf
    for c in CS:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
        model.fit(X_train, y_train)
        score = metrics(y_val, model.predict_proba(X_val)[:, 1])["pr_auc"]
        if score > best_score:
            best, best_score = model, score
    return best


def fit_rdst(series: np.ndarray, y: np.ndarray) -> RandomDilatedShapeletTransform:
    return RandomDilatedShapeletTransform(
        max_shapelets=MAX_SHAPELETS,
        shapelet_lengths=SHAPELET_LENGTHS,
        proba_normalization=0.0,  # shapelets em nota/dias de verdade (não z-normalizados): dá para ler
        random_state=SEED,
        n_jobs=-1,
    ).fit(series, y)


def run_ladder(features: dict[str, np.ndarray], split: np.ndarray, y: np.ndarray):
    """Treina nos usuários de treino, escolhe C na validação, devolve P(problemático) no teste."""
    tr, va, te = (split == s for s in ("train", "val", "test"))
    rdst = fit_rdst(features["series"][tr], y[tr])
    shapelet_features = rdst.transform(features["series"])
    preds = {"B0": np.full(te.sum(), y[tr].mean())}
    fitted = {"rdst": rdst}
    for name, X in (("B1", features["B1"]), ("B2", features["B2"]), ("M", shapelet_features)):
        model = fit_logreg(X[tr], y[tr], X[va], y[va])
        preds[name] = model.predict_proba(X[te])[:, 1]
        fitted[name] = model
    return preds, fitted
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_leech.py -v`
Expected: 7 passed. (A 1ª execução compila o numba do aeon: ~30–40 s. Normal.)

- [ ] **Step 5: Commit**

```bash
git add src/fs/leech.py tests/test_leech.py
git commit -m "feat(leech): escada B0→B1→B2→M com C escolhido na validação"
```

---

### Task 5: Os top-5 shapelets

**Files:**
- Modify: `src/fs/leech.py`
- Test: `tests/test_leech.py`

**Interfaces:**
- Consumes: `rdst.shapelets_` — tupla do aeon 1.6: `(valores (n, 2, max_len), início, comprimento,
  dilatação, limiar, normalizado, médias, desvios, classe)`; coeficientes da logística do M
  (`fitted["M"][-1].coef_[0]`). Layout das features do RDST, **verificado em sonda**: shapelet *i* →
  coluna `3i` = menor distância, `3i+1` = posição, `3i+2` = nº de ocorrências.
- Produces: `top_shapelets(shapelets: tuple, coef: np.ndarray, k=5) -> pd.DataFrame` com colunas
  `shapelet, notas, dias, dilatacao, coef_distancia, coef_ocorrencias`;
  `plot_shapelets(top: pd.DataFrame, path: Path) -> None`.

- [ ] **Step 1: Escrever o teste que falha**

Acrescentar ao import: `from fs.leech import … , top_shapelets`. Acrescentar no fim:

```python
def test_top_shapelets_maps_three_columns_per_shapelet():
    values = np.zeros((3, 2, 4))
    values[1, 0, :2] = [1, 1]                   # shapelet 1: notas "1, 1"
    values[1, 1, :2] = np.log1p([3, 10])        # … com 3 e 10 dias
    shapelets = (values, np.zeros(3), np.array([4, 2, 3]), np.array([1, 2, 1]))
    coef = np.array([0.1, 0.0, 0.2,             # shapelet 0
                     -0.9, 0.0, 0.5,            # shapelet 1: a maior |coef| (distância)
                     0.0, 0.3, 0.0])            # shapelet 2
    top = top_shapelets(shapelets, coef, k=2)
    assert top["shapelet"].tolist() == [1, 2]
    first = top.iloc[0]
    assert first["notas"] == [1.0, 1.0]         # cortado no comprimento real (2), não em 4
    assert first["dias"] == [3.0, 10.0]
    assert first["dilatacao"] == 2
    assert first["coef_distancia"] == -0.9
    assert first["coef_ocorrencias"] == 0.5
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_leech.py::test_top_shapelets_maps_three_columns_per_shapelet -v`
Expected: FAIL com `ImportError: cannot import name 'top_shapelets'`.

- [ ] **Step 3: Implementar**

Em `src/fs/leech.py`, acrescentar aos imports: `from pathlib import Path` e `import matplotlib.pyplot as plt`.
No fim:

```python
def top_shapelets(shapelets: tuple, coef: np.ndarray, k: int = 5) -> pd.DataFrame:
    """Os k shapelets com maior |coeficiente| entre as 3 features de cada um.

    Leitura dos sinais (features padronizadas):
    coef_distancia < 0 → quanto MAIS PARECIDO com o padrão, MAIS risco;
    coef_ocorrencias > 0 → quanto mais vezes o padrão aparece, MAIS risco.
    """
    coef = np.asarray(coef).reshape(-1, 3)  # shapelet i → colunas 3i, 3i+1, 3i+2
    values, _, lengths, dilations = shapelets[:4]
    rows = []
    for i in np.argsort(-np.abs(coef).max(axis=1))[:k]:
        length = int(lengths[i])
        rows.append({
            "shapelet": int(i),
            "notas": values[i, 0, :length].round(2).tolist(),
            "dias": np.expm1(values[i, 1, :length]).round(1).tolist(),
            "dilatacao": int(dilations[i]),
            "coef_distancia": round(float(coef[i, 0]), 3),
            "coef_ocorrencias": round(float(coef[i, 2]), 3),
        })
    return pd.DataFrame(rows)


def plot_shapelets(top: pd.DataFrame, path: Path) -> None:
    """Uma coluna por shapelet: nota em cima, dias desde a anterior embaixo.
    Com dilatação d, os pontos do padrão estão a d revisões de distância um do outro."""
    fig, axes = plt.subplots(2, len(top), figsize=(3 * len(top), 4.5), sharey="row", squeeze=False)
    for col, (_, s) in enumerate(top.iterrows()):
        x = np.arange(len(s["notas"])) * s["dilatacao"]
        axes[0, col].plot(x, s["notas"], marker="o")
        axes[1, col].plot(x, s["dias"], marker="o", color="tab:orange")
        axes[0, col].set_title(f"#{col + 1} · dil {s['dilatacao']}\ncoef dist {s['coef_distancia']:+.2f}", fontsize=9)
        axes[1, col].set_xlabel("revisão (relativa)")
    axes[0, 0].set(ylabel="nota (1 = esqueceu)", yticks=[1, 2, 3, 4])
    axes[1, 0].set(ylabel="dias desde a anterior")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: todos passam (8 em `test_leech.py`, 7 em `test_evaluate.py`, mais os da Etapa 0).

- [ ] **Step 5: Commit**

```bash
git add src/fs/leech.py tests/test_leech.py
git commit -m "feat(leech): top-5 shapelets com mapeamento coluna→shapelet"
```

---

### Task 6: Rodada completa, relatório e README

**Files:**
- Modify: `src/fs/leech.py`, `README.md`
- Create (gerados): `reports/etapa1/README.md`, `reports/etapa1/metrics.json`,
  `reports/etapa1/top_shapelets.csv`, `reports/etapa1/{pr,calibracao,shapelets}.png`

**Interfaces:**
- Consumes: tudo das Tasks 1–5; `fs.series.L`.
- Produces: CLI `uv run python -m fs.leech`.

- [ ] **Step 1: Implementar o `main`**

Em `src/fs/leech.py`, acrescentar aos imports:

```python
import json

from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from fs.evaluate import bootstrap, ci_table, diff_ci, verdict
from fs.series import H, L
```

E no fim:

```python
CARDS = Path("data/cards.parquet")
OUT = Path("reports/etapa1")


def plot_curves(y, preds: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, p in preds.items():
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, label=name)
    ax.set(xlabel="recall", ylabel="precisão", title="Curva PR (teste)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pr.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibração perfeita")
    for name in ("B2", "M"):
        frac, mean = calibration_curve(y, preds[name], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.set(xlabel="probabilidade prevista", ylabel="fração observada", title="Calibração (teste)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "calibracao.png", dpi=150)
    plt.close(fig)


def write_markdown(principal: dict, table: pd.DataFrame, robustez: dict, top: pd.DataFrame) -> None:
    fmt = lambda r: f"{r.point:.3f} [{r.lo:.3f}, {r.hi:.3f}]"  # noqa: E731
    wide = table.assign(txt=table.apply(fmt, axis=1)).pivot(index="model", columns="metric", values="txt")
    lo, hi = principal["M_menos_B2_pr_auc"]
    lines = [
        "# Etapa 1 — Card problemático",
        "",
        f"Rótulo: ≥ {principal['L']} lapsos nos {H} dias após a 6ª revisão · "
        f"teste: {principal['n_teste']:,} cards de {principal['usuarios_teste']} usuários · "
        f"prevalência {principal['prevalencia_teste']:.1%}.",
        "",
        f"**M − B2 em PR-AUC: IC 95% [{lo:+.3f}, {hi:+.3f}] → M {principal['veredito']} B2.**",
        "",
        "| modelo | PR-AUC | ROC-AUC | Brier |",
        "|---|---|---|---|",
        *[f"| {m} | {wide.loc[m, 'pr_auc']} | {wide.loc[m, 'roc_auc']} | {wide.loc[m, 'brier']} |"
          for m in ("B0", "B1", "B2", "M")],
        "",
        "IC 95% por bootstrap de usuários do teste (1000 amostras). B0 tem PR-AUC = prevalência por construção.",
        "",
        "## Robustez ao limiar L",
        "",
        "| L | prevalência | PR-AUC B2 | PR-AUC M | IC M − B2 | veredito |",
        "|---|---|---|---|---|---|",
        *[f"| {k} | {v['prevalencia_teste']:.1%} | {v['B2']:.3f} | {v['M']:.3f} | "
          f"[{v['ic'][0]:+.3f}, {v['ic'][1]:+.3f}] | {v['veredito']} |" for k, v in robustez.items()],
        "",
        "## Top-5 shapelets",
        "",
        "![shapelets](shapelets.png)",
        "",
        top.to_markdown(index=False),
        "",
        "![curva PR](pr.png) ![calibração](calibracao.png)",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cards = pd.read_parquet(CARDS)
    check_cards(cards)
    OUT.mkdir(parents=True, exist_ok=True)
    split = cards["split"].to_numpy()
    test = split == "test"
    users = cards.loc[test, "user_id"].to_numpy()
    features = build_features(cards)

    report = {"config": {"N": N, "H": H, "L": L, "max_shapelets": MAX_SHAPELETS,
                         "shapelet_lengths": SHAPELET_LENGTHS, "Cs": list(CS), "seed": SEED},
              "robustez": {}}
    for l in (L, 1, 3):
        y = (cards["lapses_h"] >= l).to_numpy()
        preds, fitted = run_ladder(features, split, y)
        y_te = y[test]
        print(f"L={l}: prevalência no teste {y_te.mean():.1%}", flush=True)
        if l == L:
            samples = bootstrap(users, y_te, preds)
            table = ci_table(y_te, preds, samples)
            lo, hi = diff_ci(samples, "M", "B2")
            report["principal"] = {
                "L": l, "n_teste": int(test.sum()), "usuarios_teste": int(len(set(users))),
                "prevalencia_teste": float(y_te.mean()), "tabela": table.to_dict("records"),
                "M_menos_B2_pr_auc": [lo, hi], "veredito": verdict(lo, hi),
            }
            plot_curves(y_te, preds)
            top = top_shapelets(fitted["rdst"].shapelets_, fitted["M"][-1].coef_[0])
            top.to_csv(OUT / "top_shapelets.csv", index=False)
            plot_shapelets(top, OUT / "shapelets.png")
        else:
            pair = {"B2": preds["B2"], "M": preds["M"]}
            lo, hi = diff_ci(bootstrap(users, y_te, pair), "M", "B2")
            report["robustez"][f"L={l}"] = {
                "prevalencia_teste": float(y_te.mean()),
                "B2": metrics(y_te, preds["B2"])["pr_auc"], "M": metrics(y_te, preds["M"])["pr_auc"],
                "ic": [lo, hi], "veredito": verdict(lo, hi),
            }

    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    write_markdown(report["principal"], table, report["robustez"], top)
    print(f"ok: {OUT / 'README.md'} · M {report['principal']['veredito']} B2")


if __name__ == "__main__":
    main()
```

`top.to_markdown` precisa do pacote `tabulate`: acrescente `"tabulate>=0.9"` às `dependencies` do
`pyproject.toml` e rode `uv sync`.

- [ ] **Step 2: Testes continuam passando**

Run: `uv run pytest -v`
Expected: todos passam.

- [ ] **Step 3: Rodada completa, cronometrada**

Run: `time uv run python -m fs.leech`
Expected: imprime a prevalência para L = 2, 1, 3 e termina com `ok: reports/etapa1/README.md · M <veredito> B2`
em **< 30 min**. Se passar disso, **parar e reportar**.

- [ ] **Step 4: Conferir sanidade (sem mexer em nada para "melhorar" o resultado)**

Abrir `reports/etapa1/README.md` e conferir:
- PR-AUC do **B0 ≈ prevalência** do teste (por construção).
- **B1 > B0** e **M > B0**. Se M ≤ B0, é **bug**, não resultado — parar e reportar.
- Prevalência com L = 2 perto do relatório da Etapa 0.
- As figuras abrem e não estão vazias.

O veredito M × B2 é **o resultado**, seja qual for. Não mudar N, H, L, `MAX_SHAPELETS` nem os `CS`
depois de ver o teste — isso seria ajustar no teste.

- [ ] **Step 5: Traduzir os 5 shapelets (Ana, à mão)**

Para cada linha da tabela "Top-5 shapelets" em `reports/etapa1/README.md`, escrever **uma frase** em
português, por exemplo: *"esqueceu, lembrou e esqueceu de novo em menos de 1 semana → mais risco"*.
Use `coef_distancia` (< 0 = parecido com o padrão aumenta o risco) e `coef_ocorrencias`
(> 0 = aparecer mais vezes aumenta o risco). Acrescentar uma seção `## O que os shapelets dizem` no
`reports/etapa1/README.md` com as 5 frases.

- [ ] **Step 6: Atualizar o README do projeto**

No `README.md`, logo abaixo do título, na ordem da spec §7:
1. **A pergunta** em uma frase (já existe).
2. **O resultado** em uma frase, com o número e o IC de `reports/etapa1/README.md`, inclusive se for
   empate ou perda.
3. A figura `reports/etapa1/shapelets.png`.

Atualizar o **Status** para "Etapa 1 concluída" e acrescentar ao "Como reproduzir":
`4. uv run python -m fs.leech` (Etapa 1). Acrescentar a seção **Limitações** da spec §4: usuários que
abandonaram ficam de fora; card suspenso/apagado vira "sem lapsos"; prevalência varia muito entre
usuários; intervalos são decididos pelo agendador (e o dado não diz se era SM-2 ou FSRS); B2 usa FSRS
com parâmetros default.

- [ ] **Step 7: Commit**

```bash
git add src/fs/leech.py pyproject.toml uv.lock README.md reports/etapa1
git status --short   # conferir: nada de data/ nem colinha/
git commit -m "feat(leech): Etapa 1 — escada completa, IC por usuário e top-5 shapelets"
```

Depois deste commit, a Etapa 1 está concluída e **o e-mail para a Profa. Mariane pode sair** (spec §9).

---

## Fora deste plano

- Otimizar o FSRS por usuário no B2 — a spec só manda fazer se M ganhar de B2 por pouco. Se o veredito
  for "vence" com IC colado em zero, abrir isso como decisão antes de qualquer código.
- Combinar M + B2 num modelo só — "só se sobrar tempo" na spec.
- Etapa 2 (próxima revisão vs FSRS) — plano próprio; ela vai reusar `fs.evaluate`.
