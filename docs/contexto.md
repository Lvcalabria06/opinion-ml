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

No protótipo local, a pulseira recebe um aviso no início do encontro quando há alta afinidade
em pelo menos uma mídia. O aviso informa as categorias que atingiram o corte. Ao fim do
encontro, o modelo do recorte estima a chance de interação posterior no app.
O valor inicial do corte está documentado em [Match por mídia](match_por_midia.md).

- **Label `conectou`:** 1 se, em até 48 h após o fim do encontro, um dos dois seguiu, abriu o perfil,
  curtiu ou comentou uma review, ou salvou uma obra avaliada pelo outro (seção 4.4).
- **Público que importa:** **desconhecidos** (`algum_segue == 0`). Amigos interagem de qualquer jeito
  e inflam as métricas; é entre desconhecidos que o sinal agrega valor.
- **Aviso atual:** cada categoria tem seu próprio `match` e score. Uma categoria acima de 0,68 gera aviso na proximidade. O treino por mídia compara o modelo com a regra de score daquela mídia.

## Por que os dados são (ainda) simulados

Não há usuários reais do OpiniOn nem pulseira física. Usuários, obras e avaliações vêm da
**Amazon Reviews 2023** (mesmas pessoas avaliando livro, filme, jogo e música). Encontros e o
label `conectou` continuam simulados. Consequência: **qualquer métrica mede essa mistura, não
usuários reais da pulseira.** Detalhes em [`fontes_dados.md`](fontes_dados.md).

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
| Simulador e premissas | `src/opinion/gerar_bootstrap.py`, `src/opinion/mundo.py`, `config/simulacao.yaml` | [`validacao_simulacao.md`](validacao_simulacao.md) |
| Contrato da pulseira | `src/opinion/contrato.py`, `embedded/pulseira_virtual.py` | [`contrato_mqtt.md`](contrato_mqtt.md) |
| Ingestão e banco | `src/opinion/ingestao.py`, `src/opinion/db.py`, `src/opinion/carregar_historico_app.py` | [`contrato_mqtt.md`](contrato_mqtt.md) |
| Pipeline (CC2) | `src/opinion/pipeline/`, `dags/opinion_pipeline.py`, `config/pipeline.yaml` | [`dicionario_dados.md`](dicionario_dados.md) |
| Análise | `notebooks/eda_encontros.ipynb`, `src/opinion/validacao_simulacao.py` | [`validacao_simulacao.md`](validacao_simulacao.md) |

## Requisitos por sprint

O curso pede entregas de Ciência da Computação (CC) a cada sprint.

| Sprint | Requisito | Resumo |
|---|---|---|
| 1 | CC1 | dados simulados, pulseira virtual via MQTT, ingestão, banco, **Checkpoint 1** |
| 2 | CC2 | pipeline no Airflow, EDA, validação da simulação, sessão com o Design |
| 3 (S12) | CC3 | baseline explicável (Regressão Logística ou Árvore de Decisão) no **MLflow**, split temporal, comparação com a regra simples, **Checkpoint 2** |
| 4 | CC4 | comparação de quatro famílias em recortes por mídia |
| 5 | CC5 | otimização de hiperparâmetros (Optuna) e calibração |
| 6 | CC6 | validação cruzada agrupada por par, estabilidade, overfitting e janela temporal |
| 7 | CC7 | modelo integrado: consumidor MQTT, sinal de volta à pulseira, API de inferência, retreino na DAG |
| 8 | CC8 | Docker Compose com a solução completa (MLflow, API, dashboard) |
| 9 | CC9 | dashboard com tempo real, métricas e indicadores |

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
| encontro | proximidade detectada pela pulseira; ao fim, registra duração e RSSI médio |
| conexão / `conectou` | interação no app entre os dois em até 48 h após o fim do encontro |
| desconhecidos | encontro em que nenhum dos dois segue o outro (`algum_segue == 0`) |
| regra simples | ordenar encontros pelo score da mídia no respectivo recorte de treino |
| aviso por categoria | sinal da pulseira decidido pelo score e pelo match de cada mídia no início da proximidade |
| bootstrap | o dataset inicial em `data/bootstrap/`, hoje gerado pelo simulador |
| verdade da simulação | variáveis ocultas em `data/bootstrap/verdade_simulacao_NAO_USAR_COMO_FEATURE/`, só para validação |
| purga | remoção dos encontros do fim do treino cujo label só seria conhecido na validação |
| `par_id` | os dois usuários em ordem alfabética; grupo do k-fold |
| token rotativo | ID anônimo `u-…` que a pulseira anuncia; muda todo dia, só o backend traduz |
| seção X.Y | seção do documento do projeto (fora deste repositório) |
