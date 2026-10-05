# Etapa 2 — Próxima revisão · Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Como este plano é executado:** a **Ana escreve o código**; o Claude entrega uma colinha por task
> (`colinha/etapa2-task-N/`, fora do git) e revisa o código dela. **Uma task por vez**, com aprovação
> antes da próxima (a menos que a mensagem traga `--skip-validations`).

**Goal:** responder, com IC honesto, se os padrões do histórico de revisões (shapelets) acrescentam
informação ao FSRS otimizado por usuário na previsão "você vai lembrar deste card agora?".

**Architecture:** `src/fs/next_review.py` lê os revlogs (cache do HF da Etapa 0), sorteia 200 cards por
usuário e cria um ponto de previsão por revisão a partir da 2ª. Para usuários de validação e teste,
otimiza o FSRS na 1ª metade de cada um (em paralelo, com cache) e calcula P(lembrar) do FSRS default
(F0) e otimizado (F1). O M (RDST + Δt → logística) treina nos usuários de treino; o M+F é um stacking
treinado na 2ª metade da validação; tudo é avaliado na 2ª metade dos usuários de teste com
`fs.evaluate` (Etapa 1).

**Tech Stack:** Python 3.12 · pandas · aeon ≥ 1.6 · fsrs[optimizer] ≥ 6.3 · **PyTorch só-CPU** ·
scikit-learn · matplotlib · pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-forgetting-shapelets-design.md` (§5 Etapa 2, revisada em 2026-10-05)
**Depende de:** Etapa 0 (`fs.data`, `fs.series`) e Etapa 1 (`fs.evaluate`, `fs.leech.fit_logreg`,
`fs.leech.fit_rdst`, `fs.leech.BASE_DATE`).

## Global Constraints

- **Mesmos 500 usuários e mesmo split** da Etapa 0 (`sample_users`, `split_users`).
- **200 cards por usuário** (seed fixa), entre os que têm ≥ 2 revisões após colapsar o mesmo dia;
  um ponto por revisão a partir da 2ª. Rótulo: **lembrou = nota > 1**.
- Histórico: últimas **W = 8** revisões, 2 canais (nota, log(1 + dias)). Menos de **4** → completar à
  esquerda com **nota 0 e intervalo 0**. A revisão prevista **nunca** entra no próprio histórico.
- Δt (dias até a revisão prevista) entra como `log1p`, concatenado às features do RDST.
- 1ª/2ª metade = revisões do **usuário** em ordem cronológica, divididas pela contagem.
- **F0** = FSRS default. **F1** = FSRS otimizado na 1ª metade de cada usuário de val/teste, com
  **≤ 10 mil revisões de cards inteiros sorteados**, cache em `data/fsrs_params/`, otimização em
  paralelo. O FSRS sempre reproduz o histórico **completo** do card.
- **M**: treina nos usuários de treino; `C` escolhido por **log loss** na **1ª metade da validação**.
- **M+F**: logística sobre (logit p_F1, logit p_M), treinada na **2ª metade da validação**.
- Avaliação: **2ª metade dos usuários de teste**. Principal: **log loss (menor é melhor)**;
  secundárias ROC-AUC, Brier, calibração. IC 95% por bootstrap de usuários (1000).
- **M+F vence F1 só se o IC de (log loss F1 − log loss M+F) ficar inteiro acima de zero.**
- Licença: `reports/` só agregados e figuras.

**Decisões deste plano** (medidas em sondas de 2026-10-05, num projeto igual ao planejado nas Etapas
0 e 1, com dados reais):
- O RDST aceita séries de tamanhos diferentes, mas **exige toda série de treino ≥ maior shapelet (4)** —
  daí o padding até 4 (sem ele: `ValueError: Shapelet lengths array is empty`).
- O treino do M usa uma **subamostra de 150 mil pontos**, e as features vão em **float32, em pedaços**
  (~700 mil pontos × 601 features em float64 seriam ~3,4 GB). Fit do RDST com 400 mil séries: 37 s.
- PyTorch vem do índice **só-CPU** (sem isso o `uv` baixa a versão CUDA, de alguns GB).
- Otimizar o FSRS: ~56 s para 21 mil revisões num usuário real; com o limite de 10 mil e 6 processos,
  a 1ª rodada completa deve levar ~20–30 min. As seguintes leem o cache.
- Rodada real com 20 usuários (8 com FSRS otimizado): **3 min 06 s**, pico de 0,74 GB.

## Review Focus

1. **Revisão prevista vazando para o próprio histórico** → o modelo "veria a resposta". *Teste na Task 2.*
2. **Histórico com menos de 4 revisões** → completado à esquerda com zeros; sem isso, o RDST quebra no
   fit. *Teste na Task 2.*
3. **Usuário sem nenhum card com 2 revisões** → zero pontos, com as colunas certas, sem quebrar o
   `concat`. *Teste na Task 2.*
4. **FSRS otimizado com histórico cortado no meio** → o limite de 10 mil revisões pega cards inteiros.
   *Teste na Task 3.*
5. **Avaliação fora da 2ª metade do teste** (ex.: misturar pontos da 1ª metade, onde o F1 foi
   otimizado) → as previsões têm exatamente o tamanho da 2ª metade do teste. *Teste na Task 4.*

**Cuidado com o cache:** `data/fsrs_params/` e `data/etapa2_points_<n>.parquet` não sabem se você mudou
`FSRS_MAX_REVIEWS`, `CARDS_PER_USER` ou `W`. **Mudou alguma constante? Apague o cache.**

---

### Task 1: Log loss no `evaluate` e escolha de `C` por métrica

**Files:**
- Modify: `src/fs/evaluate.py`, `src/fs/leech.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `fs.evaluate.metrics`, `fs.leech.fit_logreg` (Etapa 1).
- Produces: `metrics(y, p)` passa a devolver também `"log_loss"`;
  `fit_logreg(X_train, y_train, X_val, y_val, select: str = "pr_auc") -> Pipeline` — com
  `select="log_loss"` ou `"brier"`, escolhe o `C` de **menor** valor. Chamadas da Etapa 1 continuam
  iguais (o default é `"pr_auc"`).

- [ ] **Step 1: Atualizar e escrever os testes**

Em `tests/test_evaluate.py`, **substituir** `test_metrics_perfect_predictor` (a comparação exata do
dicionário quebraria com a chave nova) e acrescentar o teste de log loss:

```python
def test_metrics_perfect_predictor():
    m = metrics([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0])
    assert (m["pr_auc"], m["roc_auc"], m["brier"]) == (1.0, 1.0, 0.0)
    assert m["log_loss"] < 1e-9


def test_log_loss_punishes_confident_mistakes():
    assert metrics([1, 0], [0.9, 0.1])["log_loss"] < metrics([1, 0], [0.6, 0.4])["log_loss"]
    assert metrics([1, 0], [0.0, 1.0])["log_loss"] > 30
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL com `KeyError: 'log_loss'`.

- [ ] **Step 3: Implementar**

Em `src/fs/evaluate.py`:
- no import do sklearn, acrescentar `log_loss`:
  `from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score`;
- no dicionário de `metrics`, depois de `"brier"`, acrescentar:
  `"log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),`
  (`clip` para errar com certeza total custar ~34,5, e não infinito; `labels` para funcionar mesmo
  numa amostra com uma classe só);
- na docstring de `diff_ci`, trocar "para Brier, o contrário" por "para Brier e log loss, o contrário".

Em `src/fs/leech.py`, substituir `fit_logreg` por:

```python
def fit_logreg(X_train, y_train, X_val, y_val, select: str = "pr_auc") -> Pipeline:
    """Escolhe C pela métrica `select` na validação (log loss e Brier: menor é melhor).
    O teste nunca participa da escolha."""
    sign = -1 if select in ("log_loss", "brier") else 1
    best, best_score = None, -np.inf
    for c in CS:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
        model.fit(X_train, y_train)
        score = sign * metrics(y_val, model.predict_proba(X_val)[:, 1])[select]
        if score > best_score:
            best, best_score = model, score
    return best
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: todos passam — inclusive os da Etapa 1 (o default `select="pr_auc"` mantém o comportamento).

- [ ] **Step 5: Commit**

```bash
git add src/fs/evaluate.py src/fs/leech.py tests/test_evaluate.py
git commit -m "feat(evaluate): log loss e escolha de C por métrica"
```

---

### Task 2: Pontos de previsão e séries

**Files:**
- Create: `src/fs/next_review.py`
- Test: `tests/test_next_review.py`

**Interfaces:**
- Consumes: `fs.series.collapse_same_day` (Etapa 0); `fs.data.load_revlog`, `sample_users`,
  `split_users` (Etapa 0); `fs.evaluate.*`, `fs.leech.BASE_DATE`, `fit_logreg`, `fit_rdst` (Etapa 1) —
  importados já aqui, usados nas próximas tasks.
- Produces:
  - constantes `CARDS_PER_USER = 200`, `W = 8`, `MIN_LEN = 4`, `FSRS_MAX_REVIEWS = 10_000`,
    `N_WORKERS = 6`, `TRAIN_POINTS = 150_000`, `SEED = 0`, `PARAMS_DIR`, `OUT`, `POINT_COLUMNS`;
  - `chronological(revlog) -> pd.DataFrame` (colapsado, cronológico, com `second_half`, `k`, `interval`);
  - `sample_cards(df, cap, seed) -> np.ndarray`;
  - `user_points(revlog, user_id, cap=200, seed=0) -> pd.DataFrame` com colunas `POINT_COLUMNS` =
    `user_id, card_id, k, delta_t, recall, second_half, h_rating, h_interval` (as duas últimas são listas);
  - `to_series_list(points) -> list[np.ndarray]`, cada série `(2, max(len, MIN_LEN))`.

- [ ] **Step 1: Escrever os testes que falham**

`tests/test_next_review.py`:

```python
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from fsrs import Card, Rating, Scheduler

from fs.leech import BASE_DATE
from fs.next_review import MIN_LEN, POINT_COLUMNS, W, to_series_list, user_points
```

```python
def revlog(rows):
    """rows: (card_id, day_offset, rating), em ordem cronológica."""
    df = pd.DataFrame(rows, columns=["card_id", "day_offset", "rating"])
    return df.assign(state=2, duration=1000, elapsed_days=0, elapsed_seconds=0)


# card 1: duas revisões no dia 0 (colapsam), depois dias 1, 3, 7 · card 2: uma revisão só, no dia 100
# ordem cronológica após colapsar: c1d0, c1d1, c1d3, c1d7, c2d100 → 2ª metade = posições 2, 3, 4
BASIC = revlog([(1, 0, 1), (1, 0, 3), (1, 1, 3), (1, 3, 1), (1, 7, 3), (2, 100, 3)])


def test_points_start_at_second_review_and_skip_single_review_cards():
    pts = user_points(BASIC, user_id=9)
    assert pts["card_id"].tolist() == [1, 1, 1]
    assert pts["k"].tolist() == [1, 2, 3]
    assert pts["recall"].tolist() == [True, False, True]   # notas 3, 1, 3
    assert pts["delta_t"].tolist() == [1, 2, 4]
    assert pts["second_half"].tolist() == [False, True, True]
    assert pts["user_id"].unique().tolist() == [9]


def test_history_is_only_the_past_and_capped_at_w():
    last = user_points(BASIC, 9).iloc[-1]
    assert list(last["h_rating"]) == [1, 3, 1]              # nunca inclui a própria revisão prevista
    assert np.allclose(last["h_interval"], np.log1p([0, 1, 2]))

    long_card = revlog([(1, d, 3) for d in range(12)] + [(2, 99, 3)])
    pts = user_points(long_card, 9)
    assert len(pts.iloc[-1]["h_rating"]) == W


def test_user_without_repeated_cards_gives_empty_points():
    pts = user_points(revlog([(1, 0, 3), (2, 5, 3)]), 9)
    assert pts.empty
    assert list(pts.columns) == POINT_COLUMNS


def test_card_sample_is_capped_and_deterministic():
    many = revlog([(c, d, 3) for c in range(50) for d in (0, 1)])
    a = user_points(many, 9, cap=10)
    assert a["card_id"].nunique() == 10
    pd.testing.assert_frame_equal(a, user_points(many, 9, cap=10))


def test_short_histories_are_left_padded_with_zeros():
    pts = pd.DataFrame({"h_rating": [[3, 1], [3] * 6], "h_interval": [[0.0, 0.7], [0.5] * 6]})
    short, full = to_series_list(pts)
    assert short.shape == (2, MIN_LEN)
    assert short[0].tolist() == [0, 0, 3, 1]
    assert short[1].tolist() == [0, 0, 0.0, 0.7]
    assert full.shape == (2, 6)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_next_review.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'fs.next_review'`.

- [ ] **Step 3: Implementar — cabeçalho do módulo**

`src/fs/next_review.py` começa assim (imports e constantes de todas as tasks da Etapa 2 de uma vez;
o PyTorch **não** é importado aqui, só dentro de `fit_user_params`, na Task 3):

```python
"""Etapa 2: os padrões do histórico de revisões têm informação que o FSRS não capta?"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fsrs import Card, Rating, ReviewLog, Scheduler
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from fs.data import load_revlog, sample_users, split_users
from fs.evaluate import bootstrap, ci_table, diff_ci, verdict
from fs.leech import BASE_DATE, fit_logreg, fit_rdst
from fs.series import collapse_same_day

CARDS_PER_USER = 200
W = 8                     # revisões de histórico que o modelo vê
MIN_LEN = 4               # = maior shapelet_length do fit_rdst: o RDST exige séries de treino desse tamanho
FSRS_MAX_REVIEWS = 10_000
N_WORKERS = 6             # processos otimizando o FSRS em paralelo (× 2 threads do torch cada)
# ponytail: subamostra do treino do M — 150 mil séries × 601 features já ocupam ~0,4 GB em float32;
# subir se a máquina tiver memória e o M ficar perto do F1
TRAIN_POINTS = 150_000
SEED = 0
PARAMS_DIR = Path("data/fsrs_params")
OUT = Path("reports/etapa2")
POINT_COLUMNS = ["user_id", "card_id", "k", "delta_t", "recall", "second_half", "h_rating", "h_interval"]
```

- [ ] **Step 4: Implementar — as funções**

```python
def chronological(revlog: pd.DataFrame) -> pd.DataFrame:
    """Uma revisão por (card, dia), em ordem cronológica, com: posição k no card, intervalo em dias
    desde a anterior do card e se a revisão cai na 1ª ou na 2ª metade do histórico do USUÁRIO."""
    df = collapse_same_day(revlog).reset_index(drop=True)
    df["second_half"] = df.index >= len(df) // 2
    df["k"] = df.groupby("card_id").cumcount()
    df["interval"] = df.groupby("card_id")["day_offset"].diff().fillna(0)
    return df


def sample_cards(df: pd.DataFrame, cap: int = CARDS_PER_USER, seed: int = SEED) -> np.ndarray:
    """Até `cap` cards com ≥ 2 revisões (só eles têm algo para prever)."""
    multi = np.sort(df.loc[df["k"] == 1, "card_id"].unique())
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(multi, size=min(cap, len(multi)), replace=False))


def user_points(revlog: pd.DataFrame, user_id: int, cap: int = CARDS_PER_USER, seed: int = SEED) -> pd.DataFrame:
    """Um ponto por revisão a partir da 2ª dos cards sorteados: histórico (≤ W) + Δt → lembrou?"""
    df = chronological(revlog)
    chosen = df[df["card_id"].isin(sample_cards(df, cap, seed))]
    rows = []
    for card_id, g in chosen.groupby("card_id", sort=True):
        ratings, days = g["rating"].to_numpy(), g["day_offset"].to_numpy()
        intervals, second = np.log1p(g["interval"].to_numpy()), g["second_half"].to_numpy()
        for k in range(1, len(g)):
            lo = max(0, k - W)
            rows.append((user_id, card_id, k, int(days[k] - days[k - 1]), bool(ratings[k] > 1),
                         bool(second[k]), ratings[lo:k].tolist(), intervals[lo:k].tolist()))
    return pd.DataFrame(rows, columns=POINT_COLUMNS)


def to_series_list(points: pd.DataFrame) -> list[np.ndarray]:
    """Uma série (2, max(len, MIN_LEN)) por ponto. Histórico curto é completado à ESQUERDA com nota 0
    e intervalo 0 — nota 0 não existe no dado, então 'histórico curto' vira um padrão legível."""
    series = []
    for r, i in zip(points["h_rating"], points["h_interval"]):
        pad = [0.0] * max(0, MIN_LEN - len(r))
        series.append(np.array([pad + list(map(float, r)), pad + list(map(float, i))]))
    return series
```

- [ ] **Step 5: Rodar e ver passar**

Run: `uv run pytest tests/test_next_review.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/fs/next_review.py tests/test_next_review.py
git commit -m "feat(next_review): pontos de previsão com histórico ≤ W e padding até 4"
```

---

### Task 3: FSRS default (F0) e otimizado por usuário (F1)

**Files:**
- Modify: `pyproject.toml`, `src/fs/next_review.py`
- Test: `tests/test_next_review.py`

**Interfaces:**
- Consumes: `chronological`, `FSRS_MAX_REVIEWS`, `PARAMS_DIR`, `N_WORKERS`, `SEED` (Task 2).
- Produces:
  - `first_half_logs(revlog, max_reviews=10_000, seed=0) -> list[fsrs.ReviewLog]`;
  - `fit_user_params(user_id: int) -> list[float]` (com cache em `data/fsrs_params/user_<id>.json`);
  - `fit_all_params(user_ids: list[int]) -> dict[int, list[float]]`;
  - `fsrs_predictions(revlog, points, parameters: list[float] | None = None) -> np.ndarray` —
    P(lembrar) em cada ponto, na mesma ordem de `points`; `None` = F0.

- [ ] **Step 1: Dependências (PyTorch só-CPU)**

Em `pyproject.toml`, na lista `dependencies`, **trocar** `"fsrs>=6.3"` por `"fsrs[optimizer]>=6.3"` e
acrescentar `"torch>=2.4"`. No fim do arquivo:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

O `torch` precisa estar listado direto (e não só vir pelo extra do fsrs) para o `uv` respeitar o
índice só-CPU.

Run: `uv sync && uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
Expected: versão terminando em `+cpu` e `False`. O `.venv` fica com ~1,5 GB.

- [ ] **Step 2: Escrever os testes que falham**

No import de `fs.next_review` em `tests/test_next_review.py`, acrescentar `first_half_logs` e
`fsrs_predictions`. Acrescentar no fim:

```python
def test_first_half_logs_keep_whole_cards_within_the_limit():
    # 3 cards × 4 revisões em dias 0..3, depois tudo de novo nos dias 10..13 (2ª metade)
    rows = [(c, d, 3) for d in range(4) for c in (1, 2, 3)] + [(c, d, 3) for d in range(10, 14) for c in (1, 2, 3)]
    logs = first_half_logs(revlog(rows), max_reviews=9)
    days = [(log.review_datetime - BASE_DATE).days for log in logs]
    assert max(days) < 10                                    # só 1ª metade
    per_card = pd.Series([log.card_id for log in logs]).value_counts()
    assert (per_card == 4).all()                             # cards inteiros
    assert len(logs) <= 9 and len(per_card) == 2


def test_fsrs_predictions_match_direct_replay():
    pts = user_points(BASIC, 9)
    got = fsrs_predictions(BASIC, pts)
    scheduler, card = Scheduler(enable_fuzzing=False), Card(card_id=1)
    expected = []
    for k, (day, r) in enumerate([(0, 1), (1, 3), (3, 1), (7, 3)]):
        when = BASE_DATE + timedelta(days=day)
        if k > 0:
            expected.append(scheduler.get_card_retrievability(card, when))
        card, _ = scheduler.review_card(card, Rating(r), review_datetime=when)
    assert got == pytest.approx(expected)
    params = list(Scheduler().parameters)
    params[0] *= 3                                           # estabilidade inicial após 'esqueci'
    custom = fsrs_predictions(BASIC, pts, parameters=params)
    assert not np.allclose(custom, got)
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_next_review.py -v`
Expected: FAIL com `ImportError: cannot import name 'first_half_logs'`.

- [ ] **Step 4: Implementar**

Acrescentar em `src/fs/next_review.py`:

```python
def first_half_logs(revlog: pd.DataFrame, max_reviews: int = FSRS_MAX_REVIEWS, seed: int = SEED) -> list[ReviewLog]:
    """Revisões da 1ª metade para otimizar o FSRS: cards INTEIROS sorteados até `max_reviews`.
    Cortar no meio do histórico de um card faria o otimizador tomar uma revisão do meio pela primeira."""
    df = chronological(revlog)
    first = df[~df["second_half"]]
    sizes = first.groupby("card_id").size()
    order = np.random.default_rng(seed).permutation(sizes.index.to_numpy())
    keep = order[np.cumsum(sizes.loc[order].to_numpy()) <= max_reviews]
    chosen = first[first["card_id"].isin(keep)]
    return [
        ReviewLog(card_id=int(c), rating=Rating(int(r)),
                  review_datetime=BASE_DATE + timedelta(days=int(d)), review_duration=None)
        for c, r, d in zip(chosen["card_id"], chosen["rating"], chosen["day_offset"])
    ]


def fit_user_params(user_id: int) -> list[float]:
    """F1: parâmetros do FSRS otimizados para um usuário, com cache em disco."""
    path = PARAMS_DIR / f"user_{user_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    import torch  # só aqui: o resto do módulo não precisa carregar o PyTorch
    from fsrs import Optimizer

    torch.set_num_threads(2)
    params = [float(x) for x in Optimizer(first_half_logs(load_revlog(user_id))).compute_optimal_parameters()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params))
    return params


def fit_all_params(user_ids: list[int]) -> dict[int, list[float]]:
    with ProcessPoolExecutor(N_WORKERS) as pool:
        return dict(zip(user_ids, pool.map(fit_user_params, user_ids)))


def fsrs_predictions(revlog: pd.DataFrame, points: pd.DataFrame, parameters: list[float] | None = None) -> np.ndarray:
    """P(lembrar) do FSRS em cada ponto, reproduzindo o histórico COMPLETO do card (não só as W últimas).
    parameters=None → parâmetros default (F0)."""
    kwargs = {} if parameters is None else {"parameters": parameters}
    scheduler = Scheduler(enable_fuzzing=False, **kwargs)
    df = chronological(revlog)
    df = df[df["card_id"].isin(points["card_id"].unique())]
    r_at = {}
    for card_id, g in df.groupby("card_id", sort=False):
        card = Card(card_id=int(card_id))
        for k, rating, day in zip(g["k"], g["rating"], g["day_offset"]):
            when = BASE_DATE + timedelta(days=int(day))
            if k > 0:
                r_at[(card_id, k)] = scheduler.get_card_retrievability(card, when)
            card, _ = scheduler.review_card(card, Rating(int(rating)), review_datetime=when)
    return np.array([r_at[(c, k)] for c, k in zip(points["card_id"], points["k"])])
```

- [ ] **Step 5: Rodar e ver passar**

Run: `uv run pytest tests/test_next_review.py -v`
Expected: 7 passed.

- [ ] **Step 6: Conferir o otimizador num usuário real** (não tem teste unitário: precisa de dado e leva ~1 min)

Run:

```bash
uv run python -c "
import time; from fs.next_review import fit_user_params
t = time.time(); p = fit_user_params(4507); print(f'{time.time()-t:.0f}s', [round(x, 3) for x in p[:4]])
t = time.time(); fit_user_params(4507); print(f'cache: {time.time()-t:.2f}s')"
```

Expected: a 1ª linha leva dezenas de segundos e imprime 4 parâmetros; a 2ª (cache) leva < 0,1 s.
Existe `data/fsrs_params/user_4507.json`. Apague esse arquivo depois (`rm data/fsrs_params/user_4507.json`):
o usuário 4507 pode nem estar entre os 500 sorteados.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/fs/next_review.py tests/test_next_review.py
git status --short   # nada de data/
git commit -m "feat(next_review): FSRS default e otimizado por usuário, com cache"
```

---

### Task 4: Modelos — M e o stacking M+F

**Files:**
- Modify: `src/fs/next_review.py`
- Test: `tests/test_next_review.py`

**Interfaces:**
- Consumes: `to_series_list` (Task 2); `fit_rdst`, `fit_logreg(…, select="log_loss")` (Etapa 1 + Task 1).
- Produces: `logit(p) -> np.ndarray`; `shapelet_features(rdst, series, delta_t, chunk=50_000) -> np.ndarray`
  (float32); `run_models(points) -> tuple[dict[str, np.ndarray], LogisticRegression]` — o dict tem
  `F0, F1, M, M+F` com P(lembrar) **só nos pontos da 2ª metade dos usuários de teste**, na ordem de
  `points`. `points` precisa das colunas `POINT_COLUMNS + ["split", "p_f0", "p_f1"]`.

- [ ] **Step 1: Escrever o teste que falha**

Em `tests/test_next_review.py`, acrescentar `from fs.evaluate import metrics` aos imports e `run_models`
ao import de `fs.next_review`. Acrescentar no fim:

```python
def planted_points(n_users=40, per_user=150, seed=0):
    """Lembrar depende de um padrão no histórico (dois 'esqueci' seguidos no fim);
    o 'FSRS' falso (p_f1) não sabe disso."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        split = "train" if u < 20 else "val" if u < 30 else "test"
        for j in range(per_user):
            hist = rng.integers(2, 5, rng.integers(1, W + 1)).tolist()
            bad = len(hist) >= 2 and rng.random() < 0.4
            if bad:
                hist[-2:] = [1, 1]
            recall = rng.random() < (0.3 if bad else 0.9)
            rows.append((u, j, len(hist), int(rng.integers(1, 30)), recall, j >= per_user // 2,
                         hist, np.log1p(rng.integers(1, 30, len(hist))).tolist(), split, 0.75, 0.75))
    return pd.DataFrame(rows, columns=["user_id", "card_id", "k", "delta_t", "recall", "second_half",
                                       "h_rating", "h_interval", "split", "p_f0", "p_f1"])


def test_stacking_uses_shapelets_when_fsrs_misses_the_pattern():
    points = planted_points()
    preds, stack = run_models(points)
    test = ((points["split"] == "test") & points["second_half"]).to_numpy()
    y = points.loc[test, "recall"].to_numpy()
    ll = {name: metrics(y, p)["log_loss"] for name, p in preds.items()}
    assert ll["M+F"] < ll["F1"] - 0.05
    assert stack.coef_[0][1] > 0.5                           # o stacking dá peso ao M
    assert set(preds) == {"F0", "F1", "M", "M+F"}
    assert all(len(p) == test.sum() for p in preds.values())
```

O "FSRS" falso deste teste dá sempre 0,75: não sabe nada. Se o stacking estiver ligado direito, ele
tem que descobrir o padrão pelo M e dar peso a ele.

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_next_review.py -v`
Expected: FAIL com `ImportError: cannot import name 'run_models'`.

- [ ] **Step 3: Implementar**

Acrescentar em `src/fs/next_review.py`:

```python
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def shapelet_features(rdst, series: list[np.ndarray], delta_t: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """RDST + log(1 + Δt), em pedaços e em float32 para caber na memória."""
    blocks = [rdst.transform(series[s:s + chunk]).astype(np.float32) for s in range(0, len(series), chunk)]
    return np.hstack([np.vstack(blocks), np.log1p(delta_t)[:, None].astype(np.float32)])


def run_models(points: pd.DataFrame) -> tuple[dict[str, np.ndarray], LogisticRegression]:
    """F0, F1, M e M+F, devolvendo P(lembrar) nos pontos de AVALIAÇÃO (2ª metade dos usuários de teste)."""
    split, second, y = points["split"].to_numpy(), points["second_half"].to_numpy(), points["recall"].to_numpy()
    rng = np.random.default_rng(SEED)
    train = np.flatnonzero(split == "train")
    train = np.sort(rng.choice(train, size=min(TRAIN_POINTS, len(train)), replace=False))
    val1 = np.flatnonzero((split == "val") & ~second)   # escolhe o C do M
    val2 = np.flatnonzero((split == "val") & second)    # treina o stacking M+F
    test = np.flatnonzero((split == "test") & second)   # avaliação

    series = to_series_list(points)
    delta_t = points["delta_t"].to_numpy(float)
    rdst = fit_rdst([series[i] for i in train], y[train])
    feats = {name: shapelet_features(rdst, [series[i] for i in idx], delta_t[idx])
             for name, idx in (("train", train), ("val1", val1), ("val2", val2), ("test", test))}
    m = fit_logreg(feats["train"], y[train], feats["val1"], y[val1], select="log_loss")

    p_f1 = points["p_f1"].to_numpy()
    p_m_val2, p_m_test = m.predict_proba(feats["val2"])[:, 1], m.predict_proba(feats["test"])[:, 1]
    stack = LogisticRegression().fit(np.column_stack([logit(p_f1[val2]), logit(p_m_val2)]), y[val2])
    p_mf = stack.predict_proba(np.column_stack([logit(p_f1[test]), logit(p_m_test)]))[:, 1]
    preds = {"F0": points["p_f0"].to_numpy()[test], "F1": p_f1[test], "M": p_m_test, "M+F": p_mf}
    return preds, stack
```

Por que três fatias da validação/teste: `val1` (1ª metade da validação) escolhe o `C` do M;
`val2` (2ª metade) treina o stacking com previsões que o M **não viu** no treino nem na escolha do
`C`; `test` (2ª metade do teste) só avalia. Nenhum dado decide duas coisas.

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: todos passam (8 em `test_next_review.py`).

- [ ] **Step 5: Commit**

```bash
git add src/fs/next_review.py tests/test_next_review.py
git commit -m "feat(next_review): M (RDST + Δt) e stacking M+F"
```

---

### Task 5: Pipeline, rodada completa e relatório

**Files:**
- Modify: `src/fs/next_review.py`, `README.md`
- Create (gerados): `reports/etapa2/README.md`, `reports/etapa2/metrics.json`, `reports/etapa2/calibracao.png`

**Interfaces:**
- Consumes: tudo das Tasks 1–4.
- Produces: `build_points(n_users) -> pd.DataFrame` (cache em `data/etapa2_points_<n>.parquet`);
  `write_report(points, preds, stack) -> None`; CLI `uv run python -m fs.next_review [--n-users N]`.

- [ ] **Step 1: Implementar**

Acrescentar em `src/fs/next_review.py`:

```python
def build_points(n_users: int) -> pd.DataFrame:
    """Pontos de todos os usuários, com p_f0/p_f1 para validação e teste. Cache em data/ (apague para refazer)."""
    cache = Path(f"data/etapa2_points_{n_users}.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    users = sample_users(n_users)
    split = split_users(users)
    evaluated = [u for u in users if split[u] != "train"]
    print(f"otimizando o FSRS de {len(evaluated)} usuários (cache em {PARAMS_DIR})…", flush=True)
    params = fit_all_params(evaluated)
    parts = []
    for i, u in enumerate(users, 1):
        revlog = load_revlog(u)
        pts = user_points(revlog, u)
        pts["split"] = split[u]
        if split[u] != "train" and len(pts):
            pts["p_f0"] = fsrs_predictions(revlog, pts)
            pts["p_f1"] = fsrs_predictions(revlog, pts, params[u])
        parts.append(pts)
        print(f"[{i}/{len(users)}] user {u} ({split[u]}): {len(pts)} pontos", flush=True)
    points = pd.concat([p for p in parts if len(p)], ignore_index=True)
    cache.parent.mkdir(exist_ok=True)
    points.to_parquet(cache, index=False)
    return points


def write_report(points: pd.DataFrame, preds: dict, stack: LogisticRegression) -> None:
    test = ((points["split"] == "test") & points["second_half"]).to_numpy()
    users, y = points.loc[test, "user_id"].to_numpy(), points.loc[test, "recall"].to_numpy()
    samples = bootstrap(users, y, preds)
    table = ci_table(y, preds, samples)
    lo, hi = diff_ci(samples, "F1", "M+F", "log_loss")      # F1 − M+F: positivo = M+F erra menos
    m_lo, m_hi = diff_ci(samples, "F1", "M", "log_loss")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibração perfeita")
    for name, p in preds.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.set(xlabel="P(lembrar) prevista", ylabel="fração que lembrou", title="Calibração (teste)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "calibracao.png", dpi=150)
    plt.close(fig)

    fmt = lambda r: f"{r.point:.4f} [{r.lo:.4f}, {r.hi:.4f}]"  # noqa: E731
    wide = table.assign(txt=table.apply(fmt, axis=1)).pivot(index="model", columns="metric", values="txt")
    w_f1, w_m = stack.coef_[0]
    lines = [
        "# Etapa 2 — Próxima revisão",
        "",
        f"Avaliação: 2ª metade dos usuários de teste · {test.sum():,} revisões de {len(set(users))} usuários · "
        f"lembrou em {y.mean():.1%}.",
        "",
        f"**F1 − M+F em log loss: IC 95% [{lo:+.4f}, {hi:+.4f}] → M+F {verdict(lo, hi)} F1.**",
        f"F1 − M em log loss: IC 95% [{m_lo:+.4f}, {m_hi:+.4f}] → M {verdict(m_lo, m_hi)} F1.",
        "",
        "| modelo | log loss ↓ | ROC-AUC ↑ | Brier ↓ |",
        "|---|---|---|---|",
        *[f"| {m} | {wide.loc[m, 'log_loss']} | {wide.loc[m, 'roc_auc']} | {wide.loc[m, 'brier']} |"
          for m in ("F0", "F1", "M", "M+F")],
        "",
        "IC 95% por bootstrap de usuários do teste (1000 amostras).",
        "",
        f"Pesos do stacking M+F (sobre o logit de cada um): F1 {w_f1:+.3f} · M {w_m:+.3f}. "
        "Peso de M perto de zero = os shapelets não acrescentam nada ao FSRS.",
        "",
        "![calibração](calibracao.png)",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    report = {"n_teste": int(test.sum()), "tabela": table.to_dict("records"),
              "F1_menos_MF_log_loss": [lo, hi], "veredito_MF": verdict(lo, hi),
              "F1_menos_M_log_loss": [m_lo, m_hi], "veredito_M": verdict(m_lo, m_hi),
              "stacking": {"F1": float(w_f1), "M": float(w_m)}}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=500)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    points = build_points(args.n_users)
    preds, stack = run_models(points)
    write_report(points, preds, stack)
    print(f"ok: {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test com 20 usuários**

Run: `time uv run python -m fs.next_review --n-users 20`
Expected: "otimizando o FSRS de 8 usuários…", 20 linhas `[i/20] user …`, `ok: reports/etapa2/README.md`,
em poucos minutos (na sonda: 3 min 06 s). Abrir o README gerado e conferir que a tabela tem F0, F1, M
e M+F.

Depois, **apagar o resultado do smoke** para ele não se misturar com o de verdade:
`rm -r reports/etapa2 data/etapa2_points_20.parquet`. Os parâmetros em `data/fsrs_params/` podem
ficar: são de usuários sorteados por `sample_users(20)`, e serão reusados se aparecerem de novo.

- [ ] **Step 3: Rodada completa (500 usuários), em segundo plano**

Run: `nohup uv run python -m fs.next_review > etapa2.log 2>&1 &` e acompanhar com `tail -f etapa2.log`.
Expected: a otimização do FSRS de 200 usuários leva ~20–30 min na 1ª vez (6 em paralelo); o resto,
poucos minutos. Termina com `ok: reports/etapa2/README.md`. Se a memória apertar (o pico deve ficar
em poucos GB), **parar e reportar** — a saída é reduzir `TRAIN_POINTS`, e isso é decisão da Ana.
Depois apagar o `etapa2.log`.

- [ ] **Step 4: Conferir sanidade (sem mexer em nada para "melhorar" o resultado)**

Abrir `reports/etapa2/README.md`:
- **"lembrou em"** deve ficar perto de 85–90% (o Anki mira ~90% de retenção).
- **F1 ≤ F0 em log loss** (otimizar não pode piorar muito; se piorar, é bug no `first_half_logs` ou na
  ordem cronológica — parar e reportar).
- **M+F ≤ F1 + um fio** em log loss: o stacking pode, no pior caso, dar peso ~0 ao M.
- A figura de calibração abre.

O veredito M+F × F1 é **o resultado**, seja qual for. A spec já declarou a expectativa de que o M
sozinho perde para o F1.

- [ ] **Step 5: README do projeto**

No `README.md`, acrescentar uma seção **"Etapa 2 — próxima revisão"** com:
1. A pergunta em uma frase ("os padrões do histórico acrescentam algo ao FSRS otimizado?").
2. O resultado em uma frase, com o IC de `reports/etapa2/README.md` e os pesos do stacking.
3. Link para `reports/etapa2/README.md`.
4. Referência para o leitor comparar: os números publicados do
   [srs-benchmark](https://github.com/open-spaced-repetition/srs-benchmark) (HLR, SM-2, redes neurais) —
   **citados, não reimplementados**, e com o aviso de que o protocolo deles é diferente do nosso.

Em "Como reproduzir", acrescentar `5. uv run python -m fs.next_review` (Etapa 2; a 1ª vez otimiza o
FSRS de 200 usuários e leva ~30 min). Em **Limitações**, acrescentar as duas da spec §5: F1 otimizado
com no máximo 10 mil revisões por usuário; pontos de 200 cards por usuário.

- [ ] **Step 6: Commit**

```bash
git add src/fs/next_review.py README.md reports/etapa2
git status --short   # nada de data/, colinha/ nem etapa2.log
git commit -m "feat(next_review): Etapa 2 — M e M+F contra FSRS otimizado, com IC por usuário"
```

---

## Fora deste plano

- Otimizar o F1 sem o limite de 10 mil revisões — a spec escolheu limitar. Se o veredito for "vence"
  com IC colado em zero, reabrir como decisão: um F1 mais forte poderia desfazer a vitória.
- Treinar o M+F com as 600 features do RDST (a versão original da spec) — trocado pelo stacking.
- Interpretar os shapelets da Etapa 2 — a spec só pede interpretação na Etapa 1.
