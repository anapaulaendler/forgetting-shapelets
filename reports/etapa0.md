etapa 0: base comum

gerado com `uv run python -m fs.data --n-users 5` em 0.0 min
dataset `open-spaced-repetition/anki-revlogs-10k` @ `75299740cff0` . seed 2903 . N = 6, H = 180 dias, L = 2, teto = 200.

usuarios por split (com algum card) => treino 3 . val 1 . teste 1
usuarios sem nenhum card elegivel => 0
revisoes brutas baixadas => 240,853
cards elegiveis antes do teto => 12,104 (top-2 usuarios: 67%)
cards elegiveis por usuario => mediana 3417 . p90 4126 . max 4443
cards apos o teto => 1,000
prevalencia de card problematico no treino (L = 2) => 6.0%
robustez no treino: L = 1 . L = 3 => 15.5% . 3.3%

![revisões por card](etapa0/n_reviews.png)

![séries de exemplo](etapa0/series_exemplo.png)
