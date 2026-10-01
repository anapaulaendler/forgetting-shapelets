# forgetting-shapelets — explicado como se fosse para uma criança

> Versão simples da [spec técnica](2026-10-01-forgetting-shapelets-design.md).
> Se as duas discordarem, vale a técnica.

## A história

Imagina que você está estudando com **cartinhas**. Na frente tem uma pergunta, atrás a resposta.
Um aplicativo chamado **Anki** mostra as cartinhas de tempos em tempos, e a cada vez você diz
como foi:

- 😣 **"Esqueci"** (nota 1)
- 😐 "Foi difícil" (nota 2)
- 🙂 "Lembrei" (nota 3)
- 😎 "Fácil demais" (nota 4)

Se você lembra, o aplicativo espera **mais tempo** antes de mostrar de novo. Se você esquece,
ele mostra de novo **logo**.

Algumas cartinhas são **teimosas**: você esquece, esquece, esquece de novo. A gente chama essas
de **cartinhas problemáticas**.

## A pergunta

> Dá para descobrir **cedo** quais cartinhas vão ser teimosas?

É como um médico que olha os primeiros sintomas e já desconfia do que vem pela frente. Se o
aplicativo soubesse cedo, poderia ajudar com aquela cartinha antes de você perder tempo com ela.

## Os três números mágicos: N, K e L

Pensa na vida de uma cartinha como uma fila de vezes em que ela apareceu:

```
vezes:   1  2  3  4  5  6 │ 7  8  9  10 11 12 13 14 15 16
         └─ N = 6 ───────┘ │ └────────── K = 10 ──────────┘
         o computador VÊ    │ o computador NÃO vê — é aqui
                            │ que a gente confere a resposta
```

- **N = 6:** o computador só pode olhar as **6 primeiras vezes**. É como espiar o começo do filme.
- **K = 10:** depois, a gente olha as **10 vezes seguintes** para saber o que aconteceu de verdade.
  O computador **nunca** espia essa parte.
- **L = 3:** se nessas 10 vezes você **esqueceu 3 ou mais**, a cartinha era teimosa.

**Por que 3?** Uma cartinha normal você esquece mais ou menos **1 vez a cada 10**. Esquecer 3
vezes por puro azar é raro (acontece umas 7 vezes em 100). Então 3 é um bom sinal de "essa
cartinha é diferente".

## O que é um shapelet?

Sabe quando você ouve **só três notinhas** de uma música e já sabe qual é? Você não precisa da
música inteira; um **pedacinho** basta.

Um **shapelet** é isso: um **pedacinho de padrão** que aparece muito nas cartinhas teimosas.
Por exemplo: *"esqueceu, lembrou, e logo depois esqueceu de novo"*. O computador procura esses
pedacinhos sozinho e depois mostra para a gente quais encontrou.

A parte legal é essa: o computador **não só acerta, ele mostra o porquê**. A gente consegue ler
o pedacinho e entender.

## A corrida

Para saber se os shapelets são bons mesmo, eles têm que **ganhar de outros corredores**. Cada
corredor é um pouco mais esperto que o anterior:

| Corredor | Como ele chuta |
|---|---|
| 🐢 **B0** | Chuta sempre a mesma coisa ("quase nenhuma é teimosa"). Se alguém perder dele, tem algo quebrado. |
| 🐇 **B1** | Conta coisas simples: quantas vezes você esqueceu, qual foi a última nota. |
| 🦊 **B2** | Tudo do B1, mais a opinião do **FSRS**, que é a "calculadora de memória" que o Anki já usa. |
| 🦉 **M** | Os **shapelets**, os pedacinhos de padrão. |

**Regra da vitória:** o 🦉 só ganha do 🦊 se ganhar **com folga**, não por sorte. Se empatar ou
perder, a gente **conta a verdade** no README. Perder honestamente também é um resultado.

## As regras do jogo justo

1. **Não pode espiar o futuro.** O computador só vê as 6 primeiras vezes. Um teste automático
   confere isso.
2. **Não pode decorar a pessoa.** Cada pessoa dá notas de um jeito (umas são bravas, outras
   bonzinhas). Se a mesma pessoa aparecer no treino e na prova, o computador aprende *a pessoa*,
   não *o esquecimento*. Por isso **as pessoas do treino e as da prova são diferentes**.
3. **A gente diz o quanto tem certeza.** Em vez de só "acertou 80%", a gente diz "acertou entre
   76% e 84%". Para isso, sorteamos as pessoas da prova muitas vezes e vemos o quanto o resultado
   muda.

## As etapas

| | O que acontece | Parece com… |
|---|---|---|
| **0** | Baixar as cartinhas de 500 pessoas reais e organizar tudo | Arrumar o material antes da aula |
| **1** | A corrida das cartinhas teimosas | **A prova principal.** Quando acaba, a Ana manda o e-mail para a professora. |
| **2** | Outra pergunta: "na *próxima* vez, você vai lembrar desta cartinha?" | Uma segunda prova, mais difícil |
| **3** | Uma página bonita para brincar com os pedacinhos (só se der tempo) | O mural da feira de ciências |

### A Etapa 2, um pouco mais de perto

Aqui o adversário é o **FSRS de verdade**, que foi feito justamente para essa pergunta. Ele é muito
bom, então **a gente já espera que os shapelets percam dele sozinhos**.

A pergunta esperta é outra: *se o FSRS e os shapelets trabalharem juntos, eles acertam mais que o
FSRS sozinho?* Se sim, os pedacinhos enxergam algo que o FSRS não vê. Se não, o FSRS já sabia de
tudo. As duas respostas ensinam alguma coisa.

## Coisas que podem atrapalhar (e a gente avisa)

- **Só entram cartinhas que apareceram pelo menos 16 vezes.** As que a pessoa abandonou antes
  ficam de fora, e isso pode deixar o resultado um pouco torto.
- **O aplicativo escolhe quando mostrar a cartinha.** Então o tempo entre as vezes não é "natural":
  ele já depende das notas. É como medir quanto uma criança come quando é a mãe quem serve o prato.

## O que fica pronto no fim

- Um **README** que conta, logo nas primeiras linhas, **a pergunta** e **a resposta**, com os
  números e a certeza.
- Uma **figura dos pedacinhos** que o computador encontrou, traduzidos em palavras.
- **Três comandos** para qualquer pessoa refazer tudo e ver que dá o mesmo resultado.

## Para que serve tudo isso?

Para mostrar à **Profa. Mariane** que a Ana já sabe usar shapelets em dado de verdade, com jogo
justo. O assunto do mestrado é outro (onde abrir serviços para ajudar mulheres em situação de
violência), mas a **ferramenta** é parecida, e este projeto mostra que ela sabe usar.
