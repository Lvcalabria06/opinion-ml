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
| — | Troca do dataset inventado por **datasets públicos reais** | ✅ feito | Amazon Reviews 2023, [`fontes_dados.md`](fontes_dados.md) · `data/reais/amazon2023/` |
| 3 | CC3 — baseline explicável no MLflow + Checkpoint 2 | ✅ feito | `opinion/modelo/` · `python -m opinion.checkpoint2` |
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

Regressão logística (`C=0.1`) registrada como `opinion-baseline-cc3@champion`. No teste:

| | modelo | regra simples | acaso |
|---|---|---|---|
| PR-AUC geral | 0.251 | 0.175 | 0.128 |
| PR-AUC desconhecidos | 0.185 | 0.155 | 0.077 |

```bash
python -m opinion.modelo.treinar
python -m opinion.checkpoint2          # CHECKPOINT 2 OK
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

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
| CC4 | escolha do modelo, apoiada em validação cruzada, explicabilidade e calibração |
| CC6 | k-fold **agrupado por par** (`par_id`) |
| CC7 | pontuar encontros novos (`features_encontros.parquet`) e enviar sinal pelo tópico `feedback` |
| Semana 16 | testes do protótipo do Design com usuários reais; comparar com a simulação (seção 4 da validação) |
