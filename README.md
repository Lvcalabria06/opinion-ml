# OpiniON — Ciência da Computação (Machine Learning)

| | |
|---|---|
| **Nome da solução** | **OpiniON** — pulseira que detecta encontros entre pessoas com gostos parecidos e um modelo que estima a chance de elas se conectarem no app |
| **Google Sites** | https://sites.google.com/cesar.school/opinion |
| **Relatório (template SBC)** | [`reports/relatorio_opinion.pdf`](reports/relatorio_opinion.pdf) |

**Equipe de Desenvolvimento (Ciência da Computação)**

| Nome | E-mail | GitHub |
|---|---|---|
| João Pedro | jpsa@cesar.school | [@jpp000](https://github.com/jpp000) |
| Lucas Calabria | lvc@cesar.school | [@Lvcalabria06](https://github.com/Lvcalabria06) |
| Maria Júlia Dantas | mjdma@cesar.school | [@mariajuliadantas](https://github.com/mariajuliadantas) |
| Maria Luíza Dantas | mldt@cesar.school | [@maludantass](https://github.com/maludantass) |

**Equipe de Design**

| Nome | E-mail |
|---|---|
| Fábio André Oliveira Lins | faol@cesar.school |
| Marcus Vinicius Menezes Barbosa | mvmb@cesar.school |
| Pedro Mota | pom2@cesar.school |
| Rodrigo Quirino de Mello | rqm@cesar.school |

**Aviso de afinidade por categoria.** Ao detectar proximidade, a pulseira virtual
publica `proximidade`. O backend consulta as avaliações anteriores e envia à pulseira
publicadora uma mensagem com as mídias em que o par atingiu afinidade alta. **Ao fim do
encontro**, o consumidor de inferência estima com o modelo de produção a chance de os dois
interagirem no app em até 48 h e, se ela passa do limiar, devolve o sinal à pulseira.

**Pergunta do modelo (seção 3.4):** dado que dois usuários acabaram de se encontrar, qual a chance de eles
interagirem no OpiniOn nas próximas 48 horas?

Este repositório acumula as entregas de CC sprint a sprint.

| Sprint | Requisito | O que foi entregue | Onde |
|---|---|---|---|
| 1 | **CC1** | Pulseira virtual publicando via MQTT, serviço de ingestão, banco, dataset simulado de 12 semanas, Checkpoint 1 | `src/opinion/` · `docs/contrato_mqtt.md` |
| 2 | **CC2** | Pipeline no Airflow (ingestão, compatibilidade, features, label, preparação), notebook de EDA, validação da simulação, material da sessão com o Design | `src/opinion/pipeline/` · `dags/` · `notebooks/` · `docs/validacao_simulacao.md` |
| 3 | **CC3** | Usuários/obras/avaliações da Amazon Reviews 2023; baseline (Regressão Logística) no MLflow; Checkpoint 2 | `src/opinion/fontes/` · `src/opinion/modelo/` · `docs/fontes_dados.md` |
| 4 | **CC4** | Comparação de 7 famílias no MLflow (Regressão Logística, Árvore, Random Forest, Gradient Boosting, **KNN, AdaBoost, MLP**), com matriz de confusão, ROC/AUC, precisão × recall e análise crítica | `src/opinion/modelo/comparar.py` · `docs/comparacao_modelos.md` |
| 5 | **CC5** | Otimização bayesiana (Optuna TPE) do vencedor do CC4, comparada com busca aleatória e com a grade; calibração; tudo no MLflow | `src/opinion/modelo/otimizar.py` · `docs/otimizacao.md` |
| 6 | **CC6** | Stratified k-fold agrupado por par, estabilidade, overfitting/underfitting (curva de aprendizado) e janela temporal | `src/opinion/modelo/validacao_cruzada.py` · `docs/validacao_cruzada.md` |
| 7 | **CC7** | Modelo integrado ao pipeline: consumidor MQTT pontua cada encontro novo e devolve o sinal à pulseira; API de inferência (FastAPI); retreino na DAG | `src/opinion/inferencia*.py` · `api/` · `src/opinion/modelo/producao.py` |
| 8 | **CC8** | `docker compose up` sobe tudo: MQTT, PostgreSQL, ingestão, pulseira, Airflow, MLflow, inferência, API e dashboard | `docker-compose.yml` |
| 9 | **CC9** | Dashboard (Streamlit): encontros e inferências em tempo real, sinais, métricas do modelo e indicadores (conversão, precisão dos sinais, silenciamento) | `app/` |

Usuários, obras e avaliações são **reais** (Amazon Reviews 2023). Encontros da pulseira e o label
ainda são simulados. Qualquer métrica mede essa mistura, não usuários reais da pulseira.

**Chegando agora?** Leia [`docs/contexto.md`](docs/contexto.md) (o projeto em 10 minutos) e
[`docs/status.md`](docs/status.md) (o que está feito e o que falta). Agentes de IA: [`AGENTS.md`](AGENTS.md).

---

## Visão geral

```
 Amazon Reviews 2023 (real)       CC1                                     CC2 / CC7
 fontes.amazon2023 ─► gerar_bootstrap ─► pulseira_virtual ─► MQTT ─► ingestao ─► PostgreSQL
                                              ▲   │  (encontro)                       │
                                    feedback  │   └──► inferencia_mqtt ◄── modelo ◄──┤ Airflow: opinion_pipeline
                                              └──────────────┘  │       @producao     │ ingestao → compat/features/label
                                                                ▼          ▲          │ → preparacao → treino ─► MLflow
                                                          inferencias      │
                                                                │   CC3–CC6: treinar · comparar · otimizar · validacao_cruzada
                                                                ▼
                                                   API (FastAPI) ─► Dashboard (Streamlit)   CC9
```

---

## Sprint 1 — CC1: dados e caminho até o banco

- **`src/opinion/gerar_bootstrap.py`:** simula encontros entre 1.200 usuários em 12 semanas (06/07 a 27/09/2026), em cinema, livraria, campus e convenção de games (semanas 3, 7 e 11). Gera 69.384 encontros, 38.426 interações no app, 58.251 avaliações reais, 4.443 toques de interesse e 2.343 trocas de modo. As premissas ficam em `config/simulacao.yaml`.
- **`embedded/pulseira_virtual.py`:** publica proximidade no início, encontros ao fim, toques e status no contrato MQTT da seção 4.2, com IDs anônimos que mudam todo dia. Tem dois modos: `replay` (as 12 semanas) e `ao-vivo`.
- **`src/opinion/ingestao.py`:** recebe as mensagens, valida, traduz os IDs, confere o dono da pulseira, grava no banco e envia o aviso por categoria à pulseira publicadora.
- **`src/opinion/checkpoint1.py`:** teste do Checkpoint 1 (encontro publicado → broker → banco → consulta).

## Sprint 2 — CC2: pipeline, EDA e validação

### Pipeline no Airflow (`dags/opinion_pipeline.py`)

| Task | O que faz | Código |
|---|---|---|
| `ingestao` | Lê do banco (o que o MQTT gravou + histórico do app) e grava uma "foto" em `data/staging/`. Para se faltar dado ou houver inconsistência. | `src/opinion/pipeline/ingestao.py` |
| `compatibilidade` | Similaridade de gosto por mídia entre os dois: gêneros, notas nas mesmas obras, favoritas em comum. **Só com avaliações anteriores ao encontro.** | `src/opinion/pipeline/compatibilidade.py` |
| `features` | Encontro (duração, RSSI, encontros anteriores do par), contexto (local, dia, hora, modo), perfil (atividade, conexões) e relação (já se seguem, já interagiram). **Tudo até o instante do encontro.** | `src/opinion/pipeline/features.py` |
| `label` | `conectou = 1` se houve interação entre os dois em até 48 h após o fim. Encontros com menos de 48 h ficam "aguardando". | `src/opinion/pipeline/label.py` |
| `preparacao` | Split temporal (semanas 1–8 / 9–10 / 11–12), purga da fronteira, imputação, one-hot e checagens de qualidade. Grava o dataset do modelo. | `src/opinion/pipeline/preparacao.py` |
| `resumo` | Registra no log o tamanho e a taxa de conexão de cada split. | — |

As tasks `compatibilidade`, `features` e `label` rodam em paralelo. A DAG roda diariamente e também pode ser disparada na mão.

**Saída** (`data/processed/`), que é a entrada do CC3:

| Split | Encontros | % conectou | Pares |
|---|---|---|---|
| treino (semanas 1–8) | 46.244 | 8,14% | 33.684 |
| validação (9–10) | 10.101 | 8,46% | 7.995 |
| teste (11–12) | 11.970 | 7,98% | 9.458 |

- `dataset_modelo.parquet`: 48 features, `conectou`, `split` e `par_id` (para o k-fold agrupado por par do CC6).
- `features_encontros.parquet`: todos os encontros, inclusive os que ainda aguardam as 48 h. Serve para pontuar encontros novos no CC7.
- `metadados.json`: lista de features por grupo, tamanhos, taxas e corte dos dados.

**Garantias testadas** (`tests/test_pipeline.py`, rode com `python -m pytest tests -q`):
- **Sem informação do futuro:** para 30 encontros sorteados, as features são recalculadas apagando tudo que aconteceu depois do início. O resultado tem de ser idêntico. Também conferimos que o teste falha quando um vazamento é inserido de propósito.
- **Label correto:** 100% de concordância com o gabarito da simulação, e a janela de 48 h está testada nas bordas.
- **Purga:** os 1.069 encontros do fim da semana 8, cujo label só seria conhecido na semana 9, saem do treino.

A padronização dos dados fica de propósito para o pipeline do modelo (CC3), ajustada só no treino de cada fold, para não vazar informação.

### Notebook de EDA (`notebooks/eda_encontros.ipynb`, já executado; versão HTML ao lado)

**Resultados históricos do notebook executado antes do bootstrap atual.** Os números abaixo não descrevem a rodada com avisos por categoria. Principais achados daquela rodada (entre **desconhecidos**):

| Achado | Número |
|---|---|
| Contexto importa muito | convenção 12,6% · cinema 8,9% · livraria 6,0% · campus 4,1% |
| Encontros curtos quase nunca conectam | 4% (até 1,7 min) x 13% (mais de 8 min) |
| Gosto parecido dobra a chance, mas não basta sozinho | 5,6% x 12,4%; a regra simples fica perto do acaso (PR-AUC 0,15 x 0,14) |
| Amigos dominam os positivos | 30% dos encontros, 60% das conexões |
| Perfis sociais conectam ~3× mais | gamer + gamer 13,0% x leitor + leitor 4,4% (personas provisórias) |
| Proximidade (RSSI) não muda nada entre desconhecidos | premissa da simulação a revisar |

### Validação da simulação (`docs/validacao_simulacao.md`)

Gerada por `python -m opinion.validacao_simulacao` (cerca de 1 min). Tem quatro partes:
1. **Premissas:** cada parâmetro, com valor, justificativa e como validar com dados reais.
2. **Checagens de sanidade:** na rodada atual, 15 OK, 1 atenção (RSSI) e 1 falha: a taxa de interação entre desconhecidos ficou em 1,2%, abaixo da faixa de 2%–15%.
3. **Sensibilidade:** cenários pessimista, base e otimista (`config/cenarios_sensibilidade.yaml`), rodando o mesmo pipeline e os mesmos modelos.
   - **Se mantém:** a ordem dos contextos e o melhor modelo supera a regra entre desconhecidos nos três cenários.
   - **Não se mantém:** qual algoritmo é o melhor e quais features mais pesam. Esses resultados não devem ser apresentados como descoberta.
4. **Plano de comparação com dados reais** dos testes do Design (semana 16).

### Sessão com o Design

Os achados para a sessão estão nas seções 3 a 8 e 10 do notebook (e numa apresentação de 10 slides compartilhada com a equipe). A sessão deve fechar estas decisões:
- se a pulseira deve sinalizar quem já se segue;
- a duração mínima antes do sinal;
- os limites por contexto;
- as personas do simulador;
- o custo de um sinal errado.

## Sprint 3 — CC3: dados reais e baseline no MLflow

### Dados reais (`src/opinion/fontes/amazon2023.py`)

Usuários, obras e avaliações vêm da [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
(McAuley Lab, UCSD): os mesmos usuários reais avaliando livro, filme/série, jogo e música.
Fonte, filtro e o que continua simulado: [`docs/fontes_dados.md`](docs/fontes_dados.md).

```bash
python -m opinion.fontes.amazon2023 tudo     # uma vez (baixa os brutos para data/externo/, fora do Git)
python -m opinion.gerar_bootstrap            # usa data/reais/amazon2023/ (já versionado)
```

### Baseline (`src/opinion/modelo/treinar.py`) e Checkpoint 2

Regressão Logística e Árvore de Decisão, split temporal, padronização só no treino (dentro do
`Pipeline`), comparação com a regra simples (`compat_score`). Escolha do modelo, limiar do sinal
(F0,5) e aprovação contra a regra acontecem na **validação**; o teste só é relatado. O vencedor é
registrado no MLflow como `opinion-baseline-cc3-aviso-categoria@champion`, com PR-AUC, ROC-AUC, Brier, precisão,
recall, F0,5 e matriz de confusão, mais as curvas PR, ROC e de calibração como artefatos.

O **Checkpoint 2** monta um encontro novo no formato do contrato MQTT, calcula o aviso por
categoria e estima a interação com os modelos de mídia registrados. A demonstração fica em
`data/processed/avisos_categoria.csv`. O baseline geral pode ser exibido com `--baseline` e
grava sua previsão em `data/processed/previsoes.csv`.

```bash
python -m opinion.modelo.treinar             # treina e registra
python -m opinion.checkpoint2 --usuarios usr-00694 usr-01029  # aviso musical
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # http://localhost:5000
```

## Sprint 4 — CC4: comparação de modelos

`src/opinion/modelo/comparar.py` compara **sete famílias** no mesmo protocolo do CC3 (escolha e
limiar na validação, teste medido uma vez): Regressão Logística, Árvore de Decisão, Random Forest,
Gradient Boosting e, da lista da especificação, **KNN, AdaBoost e MLP** (as grades ficam em
`config/modelo.yaml → comparacao`). Cada modelo tem um run no MLflow com métricas, matriz de confusão,
curvas ROC e precisão × recall, calibração, importância por permutação e o modelo; no run pai, os
gráficos lado a lado e o intervalo de confiança das diferenças.

Na comparação geral, **AdaBoost** foi escolhido na validação (PR-AUC entre desconhecidos 0,136 na
validação e 0,186 no teste, contra 0,131 da regra simples e 0,014 do acaso). A análise crítica está em
[`docs/comparacao_modelos.md`](docs/comparacao_modelos.md); os recortes por mídia, em
[`docs/match_por_midia.md`](docs/match_por_midia.md).

```bash
python -m opinion.modelo.comparar                  # geral: reescreve docs/comparacao_modelos.md (~8 min)
python -m opinion.modelo.comparar --midia musica   # por mídia: repita para filme, serie, livro e jogo
```

## Sprint 5 — CC5: otimização de hiperparâmetros

`src/opinion/modelo/otimizar.py` otimiza o vencedor do CC4 com **Optuna (TPE, otimização bayesiana)**,
com 30 tentativas, e roda uma **busca aleatória** com o mesmo orçamento como controle. As duas começam
pela melhor configuração da grade do CC4. O objetivo é a PR-AUC entre desconhecidos na validação; o
teste é medido uma vez. Depois da busca, as probabilidades são **calibradas** (sigmoide) para a API
mostrar a chance como probabilidade. Todas as tentativas viram runs filhos no MLflow
(`opinion-cc5-otimizacao`) e o modelo final é registrado como `opinion-cc5-otimizado@otimizado`.

Resultado: AdaBoost com `n_estimators=292, learning_rate=0,94, max_depth=1`. Na validação, a PR-AUC
subiu de 0,136 (grade) para 0,150. No teste ficou em 0,188 (IC 95% 0,146–0,248), e o ganho
sobre a grade não se confirma no teste. A calibração reduziu o Brier de 0,101 para 0,063.
Detalhes em [`docs/otimizacao.md`](docs/otimizacao.md).

```bash
python -m opinion.modelo.otimizar                  # ~15 min; --trials 10 para um teste rápido
```

## Sprint 6 — CC6: validação cruzada, estabilidade e generalização

`src/opinion/modelo/validacao_cruzada.py` avalia o modelo otimizado, a regressão logística e a regra
com **stratified k-fold agrupado por par** (`StratifiedGroupKFold`, k = 5, grupo `par_id`) nas
semanas 1–10, mede média, desvio e coeficiente de variação entre folds e compara treino com avaliação
(overfitting). Também traça uma **curva de aprendizado** e uma **janela temporal expansível**
(treina até a semana k−1 e avalia a semana k). Resultados e discussão em
[`docs/validacao_cruzada.md`](docs/validacao_cruzada.md); runs no MLflow (`opinion-cc6-validacao-cruzada`).

```bash
python -m opinion.modelo.validacao_cruzada         # ~10 min
```

## Sprint 7 — CC7: modelo integrado ao pipeline

- **Consumo automático** (`src/opinion/inferencia_mqtt.py`): assina `opinion/pulseira/+/encontro`,
  traduz os IDs rotativos, calcula as features com as **mesmas funções do pipeline** (só com o passado
  do encontro), estima a chance de interação em 48 h e grava em `inferencias`. Se a chance passa do
  limiar, publica o sinal em `opinion/pulseira/{id}/feedback` (`periferico_leve` ou
  `periferico_perceptivel`; nunca para quem está com a pulseira pausada). Contrato em
  [`docs/contrato_mqtt.md`](docs/contrato_mqtt.md).
- **API de inferência** (`api/main.py`, FastAPI, documentação em `/docs`): `POST /prever` recebe o
  encontro no formato do contrato MQTT e devolve a chance; `POST /prever/usuarios` faz o mesmo com os
  IDs reais; `GET /modelo`, `/inferencias`, `/encontros/recentes` e `/indicadores` alimentam o dashboard.
- **Retreino** (`src/opinion/modelo/producao.py`): última task da DAG. Reajusta a configuração do CC5
  com os dados novos, calibra, e só promove `opinion-producao@producao` se superar a regra simples na
  validação. API e consumidor recarregam o modelo sozinhos quando o alias muda.

```bash
uvicorn api.main:app --port 8000                   # http://localhost:8000/docs
python -m opinion.inferencia_mqtt                  # pontua os encontros que chegam pelo MQTT
python -m opinion.modelo.producao                  # (re)treina e promove o modelo de produção
```

## Sprint 8 — CC8: a solução inteira no Docker Compose

`docker compose up --build` sobe 12 serviços na ordem certa: broker, banco, carga do histórico,
ingestão, replay da pulseira, **MLflow** (servidor com proxy de artefatos), **Airflow** (a DAG roda
sozinha ao subir e termina registrando o modelo de produção), **consumidor de inferência**, pulseira
ao vivo, **API**, **dashboard** e Adminer. Ver a Opção A abaixo.

## Sprint 9 — CC9: dashboard

`app/dashboard.py` (Streamlit) lê tudo da API e se atualiza sozinho: encontros que chegam pela
ingestão, inferências e sinais emitidos, a distribuição dos níveis de sinal, as métricas de validação e
teste do modelo servido e os indicadores da seção 3.2: **taxa de conversão de encontros**, **precisão
dos sinais** (aviso de afinidade e sinal do modelo) e **taxa de silenciamento** (sinal seguido de pausa
da pulseira em até 24 h). A aba "Simular encontro" pontua um par na hora.

```bash
streamlit run app/dashboard.py                     # http://localhost:8501 (API em OPINION_API_URL)
```

---

## Como rodar

### Opção A — Docker (tudo junto)

```bash
cp .env.example .env        # uma vez: credenciais e segredo locais (o .env não vai para o Git)
docker compose up --build
```

| Serviço | Endereço |
|---|---|
| Dashboard | http://localhost:8501 |
| API de inferência (Swagger) | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Airflow (sem login, só desenvolvimento) | http://localhost:8080 → DAG `opinion_pipeline` |
| Adminer (banco) | http://localhost:8081 (PostgreSQL · servidor `postgres` · usuário, senha e base do seu `.env`) |

Sequência esperada: carga do histórico → replay da pulseira (~2–5 min) → a DAG roda sozinha
(~2 min) e registra `opinion-producao@producao` no MLflow → o serviço `inferencia` carrega o modelo e
passa a pontuar os encontros ao vivo (`docker compose logs -f inferencia`) → o dashboard mostra tudo.
Até o primeiro treino terminar, `/prever` responde 503.

```bash
docker compose exec ingestao python -m opinion.checkpoint1         # Checkpoint 1
docker compose exec airflow airflow dags test opinion_pipeline     # pipeline + treino na mão
docker compose logs -f inferencia                                  # encontros ao vivo sendo pontuados
```

No **Linux**, coloque o seu `id -u` em `AIRFLOW_UID` no `.env`, para o Airflow conseguir escrever em `./data`.

### Opção B — Python local (o caminho testado)

**Já tinha o projeto antes da reorganização das pastas (D-012)?** Rode uma vez, da raiz, antes de
treinar: `pip install -r requirements-analise.txt` (agora instala o pacote de `src/`) e
`mv mlflow.db mlruns/` (o MLflow passou a guardar o banco em `mlruns/`; sem isso os modelos já
registrados somem). Se sobrarem na raiz as pastas antigas `opinion` ou `airflow` só com cache
(`__pycache__`, logs), pode apagá-las.

```bash
pip install -r requirements-analise.txt
cp .env.example .env                             # configuração local (uma vez)
# terminal 1: broker MQTT
docker run -p 1883:1883 eclipse-mosquitto:2 mosquitto -c /mosquitto-no-auth.conf

# CC1 — dados chegando pelo MQTT
python -m opinion.carregar_historico_app
python -m opinion.ingestao                       # terminal 2 (fica aberto)
python -m embedded.pulseira_virtual replay        # terminal 3 (~2 min)
python -m opinion.verificar_replay
python -m opinion.checkpoint1

# CC2 — pipeline (escolha um)
python -m opinion.pipeline.executar              # mesmas etapas da DAG, sem Airflow
airflow dags test opinion_pipeline               # com Airflow instalado (AIRFLOW__CORE__DAGS_FOLDER=dags)

python -m pytest tests -q                        # testes (inclui o de vazamento)
jupyter notebook notebooks/eda_encontros.ipynb   # EDA
python -m opinion.validacao_simulacao            # regera docs/validacao_simulacao.md

# CC3 — baseline e Checkpoint 2
python -m opinion.modelo.treinar
python -m opinion.checkpoint2 --baseline

# CC4 a CC6 — comparação, otimização e validação cruzada (registram no MLflow e reescrevem docs/)
python -m opinion.modelo.comparar
python -m opinion.modelo.otimizar
python -m opinion.modelo.validacao_cruzada

# CC7 e CC9 — modelo de produção, API, consumidor MQTT e dashboard
python -m opinion.modelo.producao
uvicorn api.main:app --port 8000                 # terminal 4
python -m opinion.inferencia_mqtt                # terminal 5
streamlit run app/dashboard.py                   # terminal 6
python -m embedded.pulseira_virtual ao-vivo --por-minuto 30   # encontros novos
```

Sem broker nem banco, o pipeline também lê direto dos arquivos: `python -m opinion.pipeline.executar --fonte arquivos`.

---

## Como testar (e o que esperar)

Siga a **Opção B** acima, nesta ordem. Cada teste prova uma parte dos requisitos.

| # | Comando | Prova | Requisito |
|---|---|---|---|
| 1 | abrir `data/bootstrap/relatorio_simulacao.md` | o dataset de 12 semanas existe e é plausível | CC1 |
| 2 | `python -m embedded.pulseira_virtual replay` | a pulseira publica via MQTT e a ingestão grava | CC1 |
| 3 | `python -m opinion.verificar_replay` | nenhuma mensagem perdida; IDs anônimos resolvidos certo | CC1 |
| 4 | `python -m opinion.checkpoint1` | **Checkpoint 1**: encontro → broker → banco → consulta | CC1 |
| 5 | `python -m embedded.pulseira_virtual ao-vivo --duracao-min 1 --duplicar 0.05 --invalidas 0.05` e depois `python -m opinion.consultar` | a ingestão ignora repetidas e rejeita defeituosas, com o motivo | CC1 |
| 6 | `python -m opinion.pipeline.executar` (ou a DAG no Airflow) | o pipeline gera o dataset do modelo | CC2 |
| 7 | `python -m pytest tests -q` | sem informação do futuro; label e janela de 48 h corretos | CC2 |
| 8 | abrir `notebooks/eda_encontros.html` | EDA e preparação dos dados com justificativas | CC2 |
| 9 | `python -m opinion.validacao_simulacao` | premissas, sanidade e sensibilidade | CC2 |
| 10 | abrir `data/reais/amazon2023/fonte.json` | usuários/obras/avaliações reais com fonte citada | CC3 |
| 11 | `python -m opinion.checkpoint2 --usuarios usr-00694 usr-01029` | **Checkpoint 2**: encontro novo recebe aviso musical e estimativa de interação | CC3 |
| 12 | `python -m opinion.modelo.comparar` e abrir `docs/comparacao_modelos.md` | 7 famílias, matriz de confusão, ROC, PR e análise crítica | CC4 |
| 13 | `python -m opinion.modelo.otimizar` e abrir o MLflow (`opinion-cc5-otimizacao`) | um run por tentativa; `opinion-cc5-otimizado@otimizado` | CC5 |
| 14 | `python -m opinion.modelo.validacao_cruzada` | k-fold agrupado por par, estabilidade, curva de aprendizado | CC6 |
| 15 | API + `python -m opinion.inferencia_mqtt` + `pulseira_virtual ao-vivo` | cada encontro novo é pontuado e o sinal volta pelo tópico `feedback` | CC7 |
| 16 | `streamlit run app/dashboard.py` | tempo real, métricas do modelo e indicadores | CC9 |

### 1. Dataset simulado
Em `data/bootstrap/relatorio_simulacao.md`, confira: 1.200 usuários, 69.384 encontros e **8,12% com `conectou = 1`**, além de picos de encontros nas semanas 3, 7 e 11 (convenção).

### 2. Replay pela pulseira virtual (~2 min)
```
[pulseira] Fim do replay: {'proximidade': 69384, 'encontro': 69384, 'interesse': 4443, 'status': 2343}
```
No terminal da ingestão aparece o progresso a cada 5 mil mensagens, com `'rejeitadas': 0`.

### 3. Nada se perdeu
```
Arquivo do bootstrap  x  banco
  encontros    69,384   69,384   OK
  interesses    4,443    4,443   OK
  status        2,343    2,343   OK

IDs rotativos traduzidos corretamente: 69,384 de 69,384 encontros
RESULTADO: tudo certo ✔
```
Se aparecer `DIFERENTE`, a ingestão ainda está processando: espere alguns segundos e rode de novo.

### 4. Checkpoint 1
```
2) Broker confirmou o recebimento (PUBACK, QoS 1): sim
4) Encontrado com a consulta: SELECT * FROM encontros WHERE encontro_id = 'e-checkpoint-...'
   usuario_a    usr-00001
   ...
CHECKPOINT 1 OK ✔
```
A mensagem sai com um ID anônimo (`u-...`) e chega ao banco já traduzida para o usuário real (`usr-00001`).

### 5. Robustez
No terminal da ingestão aparecem avisos `Rejeitada (...)`, o que é esperado. No `consultar`, as mensagens repetidas aparecem como `duplicada` e as defeituosas como rejeitadas, cada uma com o motivo.

### 6. Pipeline (CC2)
```
[ingestao       ]   ...  encontros_de 2026-07-06 ... encontros_ate 2026-09-27 ...
[compatibilidade]   ...  {'encontros': 69384, ...}
[features       ]   ...
[label          ]   ...  {'rotulados': 69384, 'pendentes': 0, 'taxa_conexao': 0.0812, ...}
[preparacao     ]   ...  {'dataset_modelo': 68315, 'encontros_purgados_fronteira': 1069, 'concordancia_label_referencia': 1.0}

Dataset do modelo: {"teste": {"encontros": 11970, ...}, "treino": {"encontros": 46244, ...}, "validacao": {"encontros": 10101, ...}}
```
Os números de encontros e pendentes variam um pouco conforme quantos encontros "ao vivo" você gerou no teste 5. Esses encontros ficam **pendentes** porque ainda não se passaram 48 h. `concordancia_label_referencia: 1.0` quer dizer que o label bate 100% com o gabarito da simulação.

**No Airflow:** com `airflow dags test opinion_pipeline`, ou o botão Trigger na interface, as 6 tasks terminam em `success`, e o log da task `resumo` mostra:
```
Dataset do modelo: 68,315 encontros | taxa de conexão 8.16%
  treino      46,244 encontros  8.14% conectou  33,684 pares
  ...
```

### 7. Testes automatizados (~1 min)
```
43 passed
```
O teste principal recalcula as features de 30 encontros **apagando tudo o que aconteceu depois do início de cada um**. Se algum valor mudar, alguma feature estaria olhando para o futuro, e o teste falha.

### 8. Notebook
Abra `notebooks/eda_encontros.html` no navegador; não precisa instalar nada. Para reexecutar, rode o teste 6 antes e depois `jupyter notebook notebooks/eda_encontros.ipynb` → *Run All*.

### 9. Validação da simulação (~1 min)
No final aparecem 15 checagens `OK`, 1 `ATENÇÃO` (RSSI), 1 `FALHA` (taxa entre desconhecidos) e a lista de conclusões que se mantêm ou não nos três cenários. O documento `docs/validacao_simulacao.md` é regerado.

### Se algo der errado

| Mensagem | Causa |
|---|---|
| `OPINION_SEGREDO_TOKEN não definida` | faltou `cp .env.example .env` |
| `Broker indisponível` | o broker MQTT (terminal 1) não está rodando |
| `Tabela usuarios vazia` | faltou `python -m opinion.carregar_historico_app` |
| Checkpoint: `o serviço de ingestão está rodando?` | o terminal da ingestão foi fechado |
| Pipeline: `nenhum encontro no banco` | rode o replay (teste 2) antes, ou use `--fonte arquivos` |

Para recomeçar do zero, pare a ingestão, apague `data/opinion.db` e repita a partir de `carregar_historico_app`.

---

## Testado / não testado

Rodada de 27/11/2026, num ambiente Linux com PostgreSQL 16, Mosquitto 2, Airflow 3.1 e um servidor
MLflow configurados **como no `docker-compose.yml`** (mesmos comandos, variáveis e portas):

- **Testado de ponta a ponta:** pulseira virtual → MQTT → ingestão → **PostgreSQL** (replay das 12
  semanas: 69.384 encontros, 0 rejeitadas, `verificar_replay` e Checkpoint 1 OK) → **Airflow**
  (`airflow dags test opinion_pipeline`: as 7 tasks em `success`, incluindo `treino`) → **servidor
  MLflow** (`opinion-producao@producao` registrado pelo Airflow com `mlflow-skinny`) → consumidor de
  inferência e **API** carregando o modelo pelo servidor → pulseira ao vivo com encontros pontuados
  automaticamente → **dashboard** (verificado com `streamlit.testing`, sem erros, inclusive a simulação).
- **Testes automatizados:** `python -m pytest tests -q` (inclui CC5, CC6, consumidor MQTT, API e indicadores).
- **Não testado:** o `docker compose up` em si (construção das imagens). O ambiente desta rodada não
  tinha acesso ao Docker Hub para baixar as imagens base; a sintaxe foi validada com
  `docker compose config`. Rodem uma vez antes da apresentação.
- Bug corrigido nesta rodada: com PostgreSQL, consultas com `%` literal (`LIKE 'e-live-%'`) quebravam
  no psycopg (`verificar_replay`); `Banco._sql` agora escapa o `%`.

## Estrutura do repositório

Segue o item 6 da especificação do projeto. As pastas extras (`config/`, `docs/`, `tests/`, `infra/`)
são ajustes permitidos pelo próprio item 6.

```
/
├── data/                  dados: bootstrap simulado (data/bootstrap/), amostra Amazon (data/reais/);
│                          staging/ e processed/ são gerados e ficam fora do Git
├── notebooks/             EDA do bootstrap e das etapas de preparação
├── src/opinion/           código de treinamento, pipeline e backend (pacote Python `opinion`)
│   ├── pipeline/          CC2: etapas do pipeline (as mesmas funções que a DAG chama)
│   ├── modelo/            CC3–CC6: baseline, comparação, otimização (Optuna), validação cruzada, produção
│   ├── fontes/            CC3: download e filtro da Amazon Reviews 2023
│   ├── gerar_bootstrap.py simulador do mundo (encontros, seguidores, interações, label)
│   ├── ingestao.py        recebe MQTT, grava encontros e envia avisos por categoria
│   ├── inferencia.py      CC7: pontua um encontro novo com o modelo de produção do MLflow
│   ├── inferencia_mqtt.py CC7: consumidor que pontua cada encontro do MQTT e devolve o sinal
│   ├── indicadores.py     CC9: conversão, precisão dos sinais e silenciamento
│   └── checkpoint1.py, checkpoint2.py
├── embedded/              simulador do sistema embarcado: a pulseira virtual (MQTT)
├── dags/                  DAG do Apache Airflow (opinion_pipeline)
├── api/                   API de inferência (CC7, FastAPI)
├── app/                   dashboard (CC9, Streamlit)
├── mlruns/                MLflow: mlflow.db e artefatos (fora do Git, só o README é versionado)
├── reports/               relatório em PDF no template SBC (fonte LaTeX em reports/latex/)
├── config/                parâmetros em YAML: simulação, fontes, pipeline, modelos
├── docs/                  contexto, decisões, status, dicionário de dados, comparação de modelos
├── tests/                 testes automáticos (pytest)
├── infra/                 Mosquitto e Dockerfile do Airflow
├── Dockerfile             imagem dos serviços Python (ingestão, pulseira, MLflow, inferência, API, dashboard)
├── docker-compose.yml     a solução inteira com um comando
├── pyproject.toml         torna o pacote de src/ instalável (pip install -e .)
├── requirements.txt       dependências dos serviços
├── requirements-ml.txt    modelo, MLflow, Optuna, API e dashboard
├── requirements-analise.txt  tudo para desenvolver (inclui o pacote)
└── README.md
```

Os comandos `python -m opinion.…` rodam da raiz depois de `pip install -r requirements-analise.txt`,
que instala o pacote de `src/` em modo editável. A pulseira roda com `python -m embedded.pulseira_virtual`.

## Pendências

Decisões com o Design, pendências de infraestrutura e próximos passos ficam em
[`docs/status.md`](docs/status.md).
