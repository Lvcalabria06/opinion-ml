# Contexto do projeto — OpiniOn, trilha de Machine Learning

Leitura de entrada para quem chega ao projeto (pessoa ou agente). O progresso fica em
[`status.md`](status.md); como rodar fica no [README](../README.md).

## O produto

O **OpiniOn** é uma plataforma social de entretenimento: as pessoas avaliam filmes, séries, livros,
jogos e músicas, seguem umas às outras e interagem pelas reviews. A equipe de **Design** está criando
uma **pulseira** que detecta quando dois usuários ficam perto um do outro (Bluetooth) em lugares como
cinema, livraria, campus e convenção de games.

## A pergunta do modelo (seção 3.4 do documento do projeto)

> Dois usuários acabaram de se encontrar. Qual a chance de eles **interagirem no app em até 48 h**?

A resposta decide se a pulseira **dá ou não um sinal** discreto para a pessoa. Um sinal errado tem
custo (incomoda, expõe); um sinal perdido também (conexão que não acontece). O custo de cada erro
ainda será fechado com o Design e vai definir a métrica e o limiar finais.

- **Label `conectou`:** 1 se, em até 48 h após o fim do encontro, um dos dois seguiu, abriu o perfil,
  curtiu ou comentou uma review, ou salvou uma obra avaliada pelo outro (seção 4.4).
- **Público que importa:** **desconhecidos** (`algum_segue == 0`). Amigos interagem de qualquer jeito
  e inflam as métricas; é entre desconhecidos que o sinal agrega valor.
- **Regra simples de referência:** sinalizar quando a compatibilidade de gosto (`compat_score`)
  passa de um limite. Todo modelo precisa superar essa regra para se justificar.

## Por que os dados são (ainda) simulados

Não há usuários reais nem pulseira física. O documento do projeto (seção 4.3) prevê dados
sintéticos, com premissas justificadas e validadas. Consequência que vale para todo o trabalho:
**qualquer métrica mede a simulação, não usuários reais.**

Em 21/10/2026 o professor avisou que dataset **inventado** perde pontos e aceitou trocar a parte de
usuários e gosto por **datasets públicos reais** de outras empresas, com fonte citada. Isso está em
andamento: ver a Frente A em [`status.md`](status.md).

## Como os dados fluem

```
 gerar_bootstrap ─► data/bootstrap/ ─► carregar_historico_app ─┐
   (simulador)          │                                     ▼
                        └─► pulseira_virtual ─► MQTT ─► ingestao ─► BANCO (SQLite local / PostgreSQL)
                                                                       │
              Airflow: opinion_pipeline                                ▼
   ingestao ─┬─ compatibilidade ─┐
             ├─ features ────────┼─ preparacao ─► data/processed/dataset_modelo.parquet ─► CC3 (modelo)
             └─ label ───────────┘
```

| Camada | Código | Documento |
|---|---|---|
| Simulador e premissas | `opinion/gerar_bootstrap.py`, `opinion/mundo.py`, `config/simulacao.yaml` | [`validacao_simulacao.md`](validacao_simulacao.md) |
| Contrato da pulseira | `opinion/contrato.py`, `opinion/pulseira_virtual.py` | [`contrato_mqtt.md`](contrato_mqtt.md) |
| Ingestão e banco | `opinion/ingestao.py`, `opinion/db.py`, `opinion/carregar_historico_app.py` | [`contrato_mqtt.md`](contrato_mqtt.md) |
| Pipeline (CC2) | `opinion/pipeline/`, `airflow/dags/opinion_pipeline.py`, `config/pipeline.yaml` | [`dicionario_dados.md`](dicionario_dados.md) |
| Análise | `notebooks/eda_encontros.ipynb`, `opinion/validacao_simulacao.py` | [`validacao_simulacao.md`](validacao_simulacao.md) |

## Requisitos por sprint

O curso pede entregas de Ciência da Computação (CC) a cada sprint.

| Sprint | Requisito | Resumo |
|---|---|---|
| 1 | CC1 | dados simulados, pulseira virtual via MQTT, ingestão, banco, **Checkpoint 1** |
| 2 | CC2 | pipeline no Airflow, EDA, validação da simulação, sessão com o Design |
| 3 (S12) | CC3 | baseline explicável (Regressão Logística ou Árvore de Decisão) no **MLflow**, split temporal, comparação com a regra simples, **Checkpoint 2** |
| depois | CC4, CC6, CC7 | escolha do modelo; k-fold agrupado por par; sinal em tempo real pela pulseira |

Semana 16: testes do protótipo do Design com pessoas reais; os encontros reais entram pelo mesmo MQTT
e pelo mesmo pipeline.

## Pessoas e papéis

- **Lucas Calabria:** fez as Sprints 1 e 2 (dono do repositório).
- **Equipe de ML (CC):** assume a partir da Sprint 3, incluindo a troca para datasets reais.
- **Design:** pulseira física, personas, padrões de sinal, testes com usuários.
- **Diego (professor):** avalia as entregas; pediu dados reais em vez de inventados.

## Glossário

| Termo | Significado |
|---|---|
| encontro | dois usuários perto um do outro por tempo suficiente, registrado pela pulseira |
| conexão / `conectou` | interação no app entre os dois em até 48 h após o fim do encontro |
| desconhecidos | encontro em que nenhum dos dois segue o outro (`algum_segue == 0`) |
| regra simples | ordenar encontros por `compat_score`; a referência que o modelo precisa superar |
| bootstrap | o dataset inicial em `data/bootstrap/`, hoje gerado pelo simulador |
| verdade da simulação | variáveis ocultas em `data/bootstrap/verdade_simulacao_NAO_USAR_COMO_FEATURE/`, só para validação |
| purga | remoção dos encontros do fim do treino cujo label só seria conhecido na validação |
| `par_id` | os dois usuários em ordem alfabética; grupo do k-fold |
| token rotativo | ID anônimo `u-…` que a pulseira anuncia; muda todo dia, só o backend traduz |
| seção X.Y | seção do documento do projeto (fora deste repositório) |
