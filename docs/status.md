# Status do projeto — onde estamos

> **Fonte única do progresso.** Atualize este arquivo sempre que uma entrega mudar de estado
> (feita, em revisão, bloqueada) ou quando uma decisão for tomada. Decisões com justificativa
> vão também para [`decisoes.md`](decisoes.md).

Última atualização: **22/10/2026** · Sprint atual: **Sprint 3 (CC3), semana S12 do cronograma**

## Resumo em uma tela

| Sprint | Requisito | Estado | Onde está |
|---|---|---|---|
| 1 | CC1 — dados simulados, pulseira virtual, MQTT, ingestão, Checkpoint 1 | ✅ feito | branch `main` |
| 2 | CC2 — pipeline no Airflow, EDA, validação da simulação | 🟡 em revisão | [PR #1](https://github.com/Lvcalabria06/opinion-ml/pull/1) (branch `sprint-2-cc2`) |
| — | Documentação de contexto, status e `AGENTS.md` | 🟡 em revisão | [PR #3](https://github.com/Lvcalabria06/opinion-ml/pull/3) (branch `docs/contexto-agentes` → `sprint-2-cc2`) |
| — | Troca do dataset inventado por **datasets públicos reais** | 🔴 a fazer (prioridade) | ver [Frente A](#frente-a--datasets-públicos-reais-prioridade) |
| 3 | CC3 — baseline explicável no MLflow + Checkpoint 2 | 🔴 a fazer | ver [Frente B](#frente-b--cc3-baseline-explicável-no-mlflow) |
| — | Testar `docker compose up` de ponta a ponta | 🔴 a fazer | ver [Pendências](#pendências-de-infraestrutura) |

Fluxo de trabalho: branches e PRs direto no repositório do Lucas (a equipe é colaboradora, sem fork).
Enquanto o PR #1 não entrar na `main`, PRs que dependem da Sprint 2 apontam para `sprint-2-cc2`.

Quem começa agora: leia [`contexto.md`](contexto.md) (10 min), depois este arquivo, depois o
[README](../README.md) para rodar.

---

## O que já está feito

### Sprint 1 — CC1 (na `main`)

- **Simulador** (`opinion/gerar_bootstrap.py`, premissas em `config/simulacao.yaml`): 1.200 usuários,
  12 semanas (06/07 a 27/09/2026), cinema, livraria, campus e convenção de games (semanas 3, 7 e 11).
  Cerca de 54,6 mil encontros, ~14% com `conectou = 1`.
- **Pulseira virtual** (`opinion/pulseira_virtual.py`): publica encontros, toques e status por MQTT no
  [contrato](contrato_mqtt.md) combinado com o Design. IDs anônimos que mudam todo dia (HMAC).
- **Ingestão** (`opinion/ingestao.py`): valida, traduz IDs, confere o dono da pulseira, grava no banco,
  rejeita mensagens com defeito (com motivo) e ignora repetidas.
- **Checkpoint 1** (`opinion/checkpoint1.py`): encontro → broker → banco → consulta.

### Sprint 2 — CC2 (no PR #1, aguardando aprovação)

- **Pipeline no Airflow** (`airflow/dags/opinion_pipeline.py` + `opinion/pipeline/`): ingestão →
  compatibilidade / features / label (em paralelo) → preparação → resumo. Saída em
  `data/processed/dataset_modelo.parquet` (33 features, `conectou`, `split`, `par_id`).
- **Split temporal:** treino semanas 1–8, validação 9–10, teste 11–12, com purga da fronteira.
- **Sem informação do futuro:** teste automático em `tests/test_pipeline.py`.
- **EDA** (`notebooks/eda_encontros.ipynb` e `.html`). Achados entre desconhecidos: convenção converte
  3× mais que campus (12,6% x 4,1%); encontros curtos quase nunca conectam; gosto parecido dobra a
  chance mas não basta sozinho; 60% das conexões vêm de quem já se seguia.
- **Validação da simulação** ([`validacao_simulacao.md`](validacao_simulacao.md)): premissas, 17
  checagens de sanidade, cenários pessimista/base/otimista.
  - Se mantém nos 3 cenários: ordem dos contextos; modelo supera a regra simples.
  - **Não** se mantém: qual algoritmo é o melhor e quais features pesam mais. Não apresente isso como
    descoberta.

---

## O que falta

### Frente A — Datasets públicos reais (prioridade)

**Origem:** mensagem do Lucas em 21/10/2026. O professor (Diego) avisou que **perderíamos pontos se
usássemos dataset inventado**. Ele aceitou a proposta de usar **dados reais de usuários publicados
por outras empresas** (ex.: Spotify para música, Netflix para filme, Steam para jogo), citando a fonte.
Fica a critério da equipe **juntar tudo num dataset só ou tratar cada mídia separadamente**.

**Tarefa:** achar datasets reais de usuários que se encaixem no treino do modelo, filtrar o que for
preciso e integrar ao repositório e ao pipeline.

**O que muda e o que fica** (análise inicial, a confirmar no desenho da Frente A):

| Parte | Hoje | Com dados reais |
|---|---|---|
| Catálogo de obras (`obras.csv`) | inventado | **real** (títulos e gêneros dos datasets) |
| Avaliações / gosto (`avaliacoes.csv`) | inventado | **real** (notas com data dos datasets) |
| Usuários (`usuarios.csv`) | inventado | derivado dos usuários reais dos datasets |
| Encontros da pulseira | simulado | continua simulado, a menos que se use um dataset real de proximidade |
| Interações no app / label | simulado | continua simulado, a menos que se use um proxy real |

Como a pulseira não existe ainda, **nenhum dataset público tem "encontro + conexão no OpiniOn"**. O
ganho dos dados reais está em tornar o gosto, a compatibilidade e a rede de amigos reais, e em reduzir
o quanto o resultado depende das premissas da equipe.

**Candidatos a avaliar** (sugestões a verificar: licença, tamanho, se tem data nas notas):

| Dataset | Mídia | Por que interessa |
|---|---|---|
| [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD) | livros, filmes/séries, jogos, música | **mesmo usuário em várias mídias**, nota 1–5 e data: evita juntar pessoas diferentes artificialmente |
| [MovieLens](https://grouplens.org/datasets/movielens/) (GroupLens) | filmes | referência acadêmica, notas com data, licença clara para pesquisa |
| [Goodreads / UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) | livros | notas e reviews com data |
| [Steam (UCSD)](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) | jogos | reviews e tempo de jogo |
| Netflix Prize (Kaggle) | filmes e séries | notas com data |
| Last.fm 1K / 360K | música | histórico de escuta com data (Spotify não publica notas de usuário) |
| [Copenhagen Networks Study](https://doi.org/10.1038/s41597-019-0325-x) | — | **proximidade Bluetooth real** entre ~700 estudantes, com RSSI e rede de amizade: o análogo mais próximo da pulseira |

**Decisões abertas da Frente A** (registrar em [`decisoes.md`](decisoes.md) quando fechadas):
1. Um dataset unificado ou um por mídia?
2. Como montar o "usuário OpiniOn": um usuário real de uma plataforma, ou o mesmo usuário em várias
   mídias (Amazon), ou combinação de usuários de plataformas diferentes (exige justificar)?
3. Os encontros continuam simulados sobre esses usuários reais, ou usamos proximidade real (Copenhagen)?
4. Onde ficam os arquivos grandes: versionados no Git (só uma amostra filtrada) ou baixados por script?

**Critério de pronto da Frente A:** o bootstrap usa obras e avaliações reais com fonte citada
(`docs/fontes_dados.md`), o pipeline do CC2 roda sem mudanças de interface, os testes passam e a
validação da simulação é regerada.

### Frente B — CC3: baseline explicável no MLflow

**Pedido (S12):** baseline explicável — **Regressão Logística ou Árvore de Decisão** — registrado no
**MLflow**, com split temporal e comparação com a **regra simples de compatibilidade**. É o
**Checkpoint 2**.

O que já existe para reaproveitar:
- Entrada pronta: `data/processed/dataset_modelo.parquet` e `metadados.json` (lista de features, coluna
  de label `conectou`, coluna de grupo `par_id`).
- Regra simples: ordenar por `compat_score` (pesos em `config/pipeline.yaml`).
- Protótipo de modelos e métricas em `opinion/validacao_simulacao.py` (funções `modelos()` e `avaliar()`).

Requisitos que o baseline precisa respeitar:
- Treinar no `treino`, escolher hiperparâmetros e limiar no `validacao`, medir **uma vez** no `teste`.
- Padronização (`StandardScaler`) **dentro** do pipeline do sklearn, ajustada só no treino.
- Métrica principal: **PR-AUC**, reportada para todos e **só para desconhecidos** (`algum_segue == 0`),
  sempre ao lado da prevalência (acaso) e da regra simples.
- Registrar no MLflow: parâmetros, métricas, lista de features, versão do dataset e o modelo no
  Model Registry. A explicação (coeficientes ou a árvore) vai como artefato.
- Variáveis de `verdade_simulacao_NAO_USAR_COMO_FEATURE/` nunca entram no modelo.

**Critério de pronto da Frente B:** um comando treina e registra o baseline; a UI do MLflow mostra o
run com PR-AUC (geral e desconhecidos) acima da regra simples no teste; README e este arquivo atualizados.

**Ordem sugerida:** desenhar a Frente A primeiro (ela muda os dados), mas escrever o código do CC3
lendo só `dataset_modelo.parquet` + `metadados.json`. Assim o baseline roda nos dados atuais e
volta a rodar sem mudança quando os dados reais entrarem.

### Para fechar com o Design

| Decisão | Proposta da equipe de ML | Impacto |
|---|---|---|
| A pulseira deve sinalizar quem eu já sigo? | não | política de sinal e talvez o label |
| Duração mínima antes do sinal | ~3 min | política de sinal |
| Personas definitivas | substituir as provisórias de `config/simulacao.yaml` | simulador |
| Custo de um sinal errado | — | define a métrica final e o limiar do modelo |
| Padrões do tópico `feedback` | — | CC7 |

### Pendências de infraestrutura

- **`docker compose up --build` nunca foi testado** (o ambiente original não tinha Docker). Alguém
  com Docker precisa rodar uma vez antes da apresentação e registrar o resultado aqui.

---

## Próximos requisitos (para planejamento)

| Requisito | O que se sabe hoje |
|---|---|
| CC4 | escolha do modelo, apoiada em validação cruzada, explicabilidade e calibração |
| CC6 | k-fold **agrupado por par** (`par_id`) |
| CC7 | pontuar encontros novos (`features_encontros.parquet`) e enviar sinal pelo tópico `feedback` |
| Semana 16 | testes do protótipo do Design com usuários reais; comparar com a simulação (seção 4 da validação) |
