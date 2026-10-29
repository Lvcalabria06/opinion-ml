# Status do projeto — onde estamos

> **Fonte única do progresso.** Atualize este arquivo sempre que uma entrega mudar de estado
> (feita, em revisão, bloqueada) ou quando uma decisão for tomada. Decisões com justificativa
> vão também para [`decisoes.md`](decisoes.md).

Última atualização: **22/10/2026** · Sprint atual: **Sprint 3 (CC3), semana S12 do cronograma**

## Resumo em uma tela

| Sprint | Requisito | Estado | Onde está |
|---|---|---|---|
| 1 | CC1 — dados simulados, pulseira virtual, MQTT, ingestão, Checkpoint 1 | ✅ feito | branch `main` |
| 2 | CC2 — pipeline no Airflow, EDA, validação da simulação | ✅ feito (EDA ainda com números da versão 100% simulada) | branch `main` |
| — | Documentação de contexto, status e `AGENTS.md` | ✅ feito | branch `main` |
| — | Troca do dataset inventado por **datasets públicos reais** | ✅ feito | Amazon Reviews 2023, [`fontes_dados.md`](fontes_dados.md) · `data/reais/amazon2023/` |
| 3 | CC3 — baseline explicável no MLflow + Checkpoint 2 | ✅ feito | branch `main` · `opinion/modelo/treinar.py` · `opinion/checkpoint2.py` |
| 4 | CC4 — comparação de 4 modelos no MLflow + análise crítica | ✅ feito | [`comparacao_modelos.md`](comparacao_modelos.md) · `opinion/modelo/comparar.py` |
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

## Sprint 3 — feito nesta branch

### Frente A — Datasets públicos reais (Amazon Reviews 2023)

**Fechado** (D-009): um dataset unificado, [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).
1.200 usuários reais, 48.018 obras, 58.251 avaliações. Encontros e label continuam simulados
(66.344 encontros, 12,3% `conectou`). Detalhes: [`fontes_dados.md`](fontes_dados.md).

```bash
python -m opinion.fontes.amazon2023 tudo     # só se data/reais/amazon2023/ não existir
python -m opinion.gerar_bootstrap
```

### Frente B — CC3: baseline no MLflow e Checkpoint 2

Regressão logística (`C=0.1`, `class_weight=balanced`) registrada como `opinion-baseline-cc3@champion`.
Escolha, limiar (F0,5) e aprovação contra a regra simples na **validação**; o teste só é relatado
(D-010). Resultado de 29/10/2026 (scikit-learn 1.9, MLflow 3.17):

| | validação | teste | regra (teste) | acaso (teste) |
|---|---|---|---|---|
| PR-AUC geral | 0.242 | 0.250 | 0.175 | 0.128 |
| PR-AUC desconhecidos | 0.174 | 0.180 | 0.155 | 0.077 |
| ROC-AUC desconhecidos | 0.709 | 0.713 | 0.669 | 0.500 |
| Brier (todos) | 0.193 | 0.203 | — | — |

No limiar (0,488), entre desconhecidos no teste: precisão 0,224, recall 0,237, F0,5 0,226, sinal
em 8,2% dos encontros (regra: precisão 0,223, recall 0,202, F0,5 0,218, sinal em 7,0%).
**Leitura honesta:** o modelo ordena melhor que a regra (PR-AUC), mas no ponto de operação a
precisão é praticamente a mesma; e com `balanced` as probabilidades ficam infladas (Brier pior
que prever a prevalência). A calibração é analisada no CC4 (abaixo).

**Checkpoint 2** (`python -m opinion.checkpoint2`): carrega o modelo do registry, confere que a
pontuação ao vivo reproduz as features do `dataset_modelo`, monta um encontro novo no formato do
contrato MQTT (IDs rotativos), passa pelo pipeline e devolve a probabilidade e o sinal (modelo e
regra). A previsão vai para `data/processed/previsoes.csv`.

```bash
python -m opinion.modelo.treinar
python -m opinion.checkpoint2          # CHECKPOINT 2 OK
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Sprint 4 — CC4: comparação de modelos

`python -m opinion.modelo.comparar` (~2 min) compara Regressão logística, Árvore de decisão,
Random Forest e Gradient Boosting no mesmo protocolo do CC3 (D-011) e escreve a análise crítica em
[`comparacao_modelos.md`](comparacao_modelos.md). Resultado de 29/10/2026, desconhecidos no teste:

| Modelo | PR-AUC (IC 95%) | Precisão no limiar | Recall | Brier |
|---|---|---|---|---|
| **Gradient Boosting** (escolhido na validação) | 0,197 (0,176–0,225) | 26,0% | 24,3% | 0,101 |
| Random Forest | 0,200 (0,177–0,227) | 27,4% | 23,1% | 0,101 |
| Regressão logística (baseline CC3) | 0,180 (0,160–0,205) | 22,4% | 23,7% | 0,203 |
| Árvore de decisão | 0,174 (0,154–0,197) | 24,3% | 16,8% | 0,103 |
| Regra simples | 0,155 (0,137–0,175) | 22,3% | 20,2% | — |

Gradient Boosting supera o baseline (+0,017, IC +0,002 a +0,033) e a regra (+0,043, IC +0,026 a
+0,061); Random Forest e Gradient Boosting são indistinguíveis. O vencedor está no MLflow como
`opinion-cc4-melhor@melhor` (ponto de partida do CC5). O Checkpoint 2 continua usando o baseline
`opinion-baseline-cc3@champion`.

**Pendente:** o planejamento diz que a previsão "aparece no dashboard". O dashboard não existe;
hoje ele seria o leitor de `previsoes.csv`. Também pendente: treino na DAG e serviço MLflow no
Docker Compose (hoje o CC3 roda só localmente).

---

## O que ainda falta

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
| CC5 | Optuna sobre o vencedor do CC4 (`opinion-cc4-melhor@melhor`, hoje Gradient Boosting) |
| CC6 | k-fold **agrupado por par** (`par_id`), estabilidade, overfitting/underfitting |
| CC7 | pontuar encontros novos (`features_encontros.parquet`) e enviar sinal pelo tópico `feedback` |
| Semana 16 | testes do protótipo do Design com usuários reais; comparar com a simulação (seção 4 da validação) |
