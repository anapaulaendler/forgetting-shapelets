# forgetting-shapelets — Design

> **Data:** 2026-10-01 · **Autora:** Ana Paula Endler · **Status:** aprovada · revisada em 2026-10-02 após sonda no dado (rótulo por horizonte em dias)
> **Definido via** `/brainstorming` (seções 1–4 aprovadas uma a uma).

## 1. Propósito

Projeto pessoal **pequeno, correto e concluído** que demonstra classificação de séries temporais
com shapelets sobre dado real. Ele é o "[projeto pessoal] … código em [link]" do e-mail para a
Profa. Mariane Cassenote (PPGInf/UFPR).

- **Não é** o tema do mestrado (que é localização de serviços da rede de atendimento a mulheres
  sob subnotificação). Este projeto é **prova de método**, não piloto da tese.
- **Não precisa** ser publicável. Precisa ser honesto: baseline forte, avaliação sem vazamento,
  resultado declarado mesmo que seja empate ou perda.
- O e-mail fala no passado ("fiz um projeto"), então **o e-mail só sai depois que a Etapa 1 fecha**.

### Critério de sucesso

- **Etapa 1 concluída** = README com pergunta, resultado com IC, figura dos shapelets interpretados,
  reproduzível em 3 comandos. Isso destrava o e-mail.
- Etapa 2 concluída = seção nova no README respondendo se shapelets acrescentam informação ao FSRS.
- "Ganhar" do baseline **não** é critério de sucesso; responder a pergunta com honestidade é.

### Restrições

- ~10h/semana; Etapa 1 cabe em ~3–4 semanas.
- Dado real e aberto; baseline honesto antes do modelo; incerteza reportada.
- Cada etapa roda em **< 30 min no notebook**. Se não couber, reduzir o nº de usuários antes de otimizar código.

## 2. Dado

**`anki-revlogs-10k`** (open-spaced-repetition, Hugging Face): logs de revisão do Anki de ~10 mil
usuários. Para cada card: sequência de revisões com nota (1 = Again, 2 = Hard, 3 = Good, 4 = Easy)
e intervalo desde a revisão anterior.

Descartados: **Duolingo HLR** (2 semanas, histórico só em contagens agregadas); **MaiMemo** (viável,
menos ferramental em volta).

Referência de protocolo e baselines: repo **`srs-benchmark`** (open-spaced-repetition).

- Subconjunto de **500 usuários** sorteados com seed fixa; sobe até 1000 só se ainda couber no
  orçamento de 30 min. Revisão do dataset no HF fixada.
- Dado bruto em `data/` (fora do git). Só relatórios e figuras são versionados.
- **Split por usuário** (treino / val / teste disjuntos). Motivo: cada pessoa tem um jeito próprio
  de dar nota; com o mesmo usuário em treino e teste, o modelo aprende o usuário, não o esquecimento.

- **Teto de 200 cards elegíveis por usuário** (sorteio com seed fixa), para nenhum usuário dominar
  treino ou métrica. Sem teto, 2 de 30 usuários concentravam 40% dos cards na sonda.

### Formato e acesso (verificado em 2026-10-02)

- Um arquivo por usuário: `revlogs/user_id=<1..10000>/data.parquet` (também `cards/` e `decks/`).
  Split por usuário = sortear quais arquivos baixar. Revisão fixada:
  `75299740cff05894ef42d7ad990666691efdd2da`. ~2 MB por usuário (~1 GB para 500).
- Colunas de `revlogs`: `card_id`, `day_offset`, `rating`, `state` (0 novo, 1 aprendendo,
  2 revisão, 3 reaprendendo), `duration`, `elapsed_days`, `elapsed_seconds`. Já vem em ordem
  cronológica. Só usuários com 5000+ revisões.
- Dataset **gated** (aceitar termos no HF + token de leitura em `~/.cache/huggingface/token`).

### Checkpoints da Etapa 0

1. ✅ **Licença:** uso permitido para estudantes e indivíduos em pesquisa própria; **proibido
   redistribuir** (só linkar). Consequência: dado nunca no git; `reports/` só com agregados e figuras.
2. ✅ **Agendador:** o dado **não indica** SM-2 vs FSRS (há `preset_id` por deck, sem o algoritmo).
   Fica só como limitação declarada.
3. ⏳ Se o **RDST do aeon aceita séries de tamanho desigual** — define o tratamento da Etapa 2.
4. ✅ **Revisões no mesmo dia:** deduplicar por `(card_id, day_offset)`, mantendo a primeira.

## 3. Etapas

| Etapa | Entrega | Gatilho |
|---|---|---|
| **0. Base comum** | Download + subconjunto + split por usuário + séries por card | — |
| **1. Card problemático** | Classificação pelas primeiras revisões, escada de baselines, shapelets interpretados, README | **E-mail sai aqui** |
| **2. Próxima revisão** | P(lembrar) revisão a revisão contra FSRS; seção nova no README | — |
| **3. Demo** (opcional) | Página estática explorando os shapelets | Só se 1 e 2 ficarem de pé |

## 4. Etapa 1 — Card problemático

### Pergunta

> Olhando só as **6 primeiras revisões** de um card, dá para prever que ele vai falhar
> **2 ou mais vezes nos 180 dias seguintes**?

### Definições (fixadas antes de olhar o teste)

- **Lapso:** nota 1 (Again).
- **N = 6:** janela de observação — as 6 primeiras revisões (após colapsar o mesmo dia). É a série
  que o modelo vê.
- **H = 180 dias:** horizonte do rótulo — as revisões com `day_offset` em (dia da 6ª, dia da 6ª + 180].
  O modelo nunca vê.
- **L = 2:** card é **"problemático"** se tiver ≥ L lapsos dentro do horizonte H.
- Revisões no mesmo dia são colapsadas em uma (fica a primeira).
- **Elegível:** card com ≥ N revisões **e** usuário ativo até pelo menos o dia da 6ª + H
  (`max(day_offset)` do usuário ≥ dia da 6ª + 180). A censura depende do **usuário**, não do card:
  um card fácil que não reaparece em 180 dias entra como não problemático, que é o que ele é.

**Por que horizonte em dias e não em nº de revisões:** a primeira versão exigia 16 revisões por
card (6 + 10 seguintes). A sonda de 2026-10-02 (30 usuários) mostrou que isso **seleciona os
difíceis**: card fácil ganha intervalo longo e nunca acumula 16 revisões. Só 19% dos cards chegavam
a 16, já com mais lapsos na janela (1,54 vs 0,90), e a prevalência ia a 24–40%. Além disso, a regra
usava o futuro: na 6ª revisão ninguém sabe se o card chega a 16.

**Por que esses valores:** com H = 180, a sonda deu 117 mil cards elegíveis em 29 de 30 usuários,
e prevalência de **7–9% com L = 2** (L = 1: ~18%; L = 3: 4–5%). L = 2 fica na faixa 5–15%, com
classe positiva suficiente. Usuário que parou antes de 180 dias sai — dado de quem abandonou o app
não ajuda a responder a pergunta. N = 6 é cedo o bastante para ser útil e longo o bastante para
ter padrão.

**Nome:** "card problemático", não "leech". O leech do Anki é ≥ 8 lapsos no total — raro e tardio.
Leech entra só como inspiração no README.

**Robustez:** repetir com L = 1 e L = 3.

### Série de entrada

2 canais × 6 passos: **nota** e **log(intervalo em dias + 1)**; a 1ª revisão tem intervalo 0.
Tamanho fixo — sem problema de série desigual nesta etapa. Na janela, "nº de lapsos" (B1) conta
toda nota 1, inclusive na fase de aprendizado.

### Escada de modelos

Cada degrau precisa vencer o anterior.

| | Modelo | Entrada |
|---|---|---|
| B0 | Prevalência (constante) | — (sanity check) |
| B1 | Regressão logística | Agregados da janela: nº de lapsos, nota média, última nota, último intervalo, crescimento do intervalo |
| B2 | Regressão logística | B1 + estado FSRS ao fim da janela (stability, difficulty), parâmetros default do pacote `fsrs` |
| **M** | Transformada **RDST** (aeon) → regressão logística | Série de 2 canais × 6 passos |

Regressão logística (e não ridge) depois do RDST porque as métricas exigem probabilidade.

Fora de escopo: FSRS otimizado por usuário nesta etapa (entra se M ganhar de B2 por pouco);
combinação M + B2 (só se sobrar tempo).

### Métricas e critério

- **Principal:** PR-AUC (classe positiva rara).
- **Secundárias:** ROC-AUC, Brier, curva de calibração.
- **IC 95%:** bootstrap **reamostrando usuários** do teste (cards do mesmo usuário são correlacionados).
- **M vence B2** só se o IC 95% da diferença de PR-AUC excluir zero. Caso contrário o README declara
  empate ou perda.
- **Independente do resultado:** top-5 shapelets traduzidos em padrão legível
  (ex.: "nota 1 → 3 com intervalo encolhendo").

### Limitações a declarar

- **Usuários que abandonaram:** só entram usuários ativos por ≥ 180 dias após a janela.
- **Card suspenso ou apagado** dentro do horizonte vira "sem lapsos" — o dado não distingue isso de
  "não precisou revisar".
- **Prevalência varia muito entre usuários** (0% a 19% com L = 2 na sonda) — reforça o split por
  usuário e o bootstrap por usuário.
- **Confusão com o agendador:** os intervalos são decididos pelo agendador, que reage às notas;
  o canal de intervalo carrega parte da informação da nota, e SM-2 vs FSRS geram séries diferentes.
- B2 usa FSRS com parâmetros default, não otimizado por usuário.

## 5. Etapa 2 — Próxima revisão

### Pergunta

> Os padrões do histórico de revisões têm informação que o FSRS não capta?

### Tarefa

Para cada revisão *i* de um card: prever **P(lembrar)** dado o histórico 1…*i−1* **e o intervalo
até a revisão *i*** (Δt, a variável mais importante — a memória decai com o tempo).

### Entrada

- Últimas **W = 8** revisões, padding à esquerda + canal-máscara ("aqui não havia revisão").
  Se o checkpoint 3 confirmar suporte a séries desiguais no RDST, usar direto e remover o padding.
- Δt entra como feature escalar concatenada às features do RDST.

### Protocolo

- Modelos globais (M, M+F) treinam nos **usuários de treino**.
- Em cada **usuário de teste**, as revisões são ordenadas no tempo: os primeiros 50% servem só para
  ajustar o FSRS por usuário (F1); **todos os modelos são avaliados nos últimos 50%**.

### Modelos

| | Modelo | O que testa |
|---|---|---|
| F0 | FSRS, parâmetros default | Baseline sem treino |
| F1 | FSRS otimizado por usuário | **Baseline forte** — o uso real |
| M | RDST(histórico) + Δt → logística | Shapelet sozinho |
| **M+F** | p_FSRS(F1) + features RDST → logística | Shapelet acrescenta informação ao FSRS? |

**Expectativa declarada:** M perde para F1 (FSRS é feito sob medida). A pergunta central é se
**M+F supera F1**. Se não superar, a conclusão é "o FSRS já extrai o que a série tem".

### Métricas e critério

- **Principal:** log loss (métrica do srs-benchmark).
- **Secundárias:** ROC-AUC, calibração.
- IC 95% por bootstrap de usuários; mesmo critério de vitória da Etapa 1.

Fora de escopo: HLR, SM-2, modelos neurais do benchmark — citados pelos números publicados no
srs-benchmark, não reimplementados.

## 6. Estrutura do repo

```
forgetting-shapelets/
├── pyproject.toml          # uv · aeon, fsrs, pandas, scikit-learn, matplotlib
├── src/fs/
│   ├── data.py             # Etapa 0: download + subconjunto + split por usuário
│   ├── series.py           # Etapa 0: revlog → séries por card
│   ├── leech.py            # Etapa 1
│   └── next_review.py      # Etapa 2
├── tests/test_series.py
├── reports/                # figuras + métricas geradas (versionadas)
└── README.md
```

Fora de escopo: notebooks como fonte da verdade (scripts geram tudo; notebook, se existir, só lê
`reports/`); config em YAML; tracking de experimento (MLflow etc.). Entram quando houver mais de um
experimento por etapa para comparar.

## 7. Entregáveis por etapa

| Etapa | Comando | Versionado em |
|---|---|---|
| 0 | `uv run python -m fs.data` | `reports/etapa0.md`: nº de usuários/cards/revisões, cards elegíveis antes/depois do teto, prevalência de card problemático (só treino), distribuição do nº de revisões, figura com 3 séries de exemplo (só figura — licença proíbe redistribuir o dado) |
| 1 | `uv run python -m fs.leech` | `reports/etapa1/`: `metrics.json`, tabela B0–M com IC, curva PR, curva de calibração, figura dos top-5 shapelets com tradução |
| 2 | `uv run python -m fs.next_review` | `reports/etapa2/`: mesmo formato, F0–M+F |

### README (ordem fixa)

1. A pergunta em uma frase.
2. O resultado em uma frase, com número e IC — inclusive se for empate ou perda.
3. Figura dos shapelets.
4. Como reproduzir (3 comandos).
5. Dado e licença · protocolo · resultados completos.
6. Limitações (§4).

## 8. Testes e tratamento de erro

`tests/test_series.py` — só onde um erro estragaria o resultado em silêncio:

- a janela nunca usa revisões além da N-ésima;
- o rótulo usa só revisões com dia em (dia da N-ésima, dia da N-ésima + H];
- revisões no mesmo dia são colapsadas;
- cards com menos de N revisões, ou de usuário inativo antes do fim do horizonte, são excluídos;
- teto de cards por usuário respeitado;
- usuários de treino, val e teste são disjuntos.

Código de modelo não tem teste unitário: a escada de baselines é o teste (M abaixo de B0 = bug).

`data.py` **aborta com mensagem clara** se o schema baixado não tiver as colunas esperadas.
Seeds fixas em todo sorteio e treino.

## 9. Ligação com o e-mail

Preencher quando a Etapa 1 fechar:

- `[nome do artigo]`: *Transfer Learning of Shapelets for Time Series Classification Using CNN*
  (Souza, Cassenote & Silva, BRACIS 2021) — **só citar depois de ler**.
- `[o que você fez]`: a frase do item 2 do README.
- `[link]`: este repo.

Ponto a levar para a conversa: shapelets classificam e precisam de rótulo; na etapa 1 da tese não
existe rótulo de "município subnotificado" — é justamente o que não se observa.
