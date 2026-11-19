# OpiniOn — Ciência da Computação (Machine Learning)

**Aviso de afinidade por categoria.** Ao detectar proximidade, a pulseira virtual
publica `proximidade`. O backend consulta as avaliações anteriores e envia à pulseira
publicadora uma mensagem com as mídias em que o par atingiu afinidade alta. Um encontro pode
render um único aviso com música e filmes. A duração e o RSSI médio chegam ao fim do encontro.

Para reproduzir o exemplo musical com o ambiente da opção B e o bootstrap versionado:

```bash
python -m opinion.pipeline.executar --fonte arquivos
python -m opinion.modelo.comparar --midia musica
python -m opinion.checkpoint2 --usuarios usr-00694 usr-01029
```

O treino por mídia usa os encontros em que ambos tinham avaliações daquela mídia, inclusive
pares de baixa afinidade. O comando aceita também `filme`, `serie`, `jogo` e `livro`.
O comparador avalia quatro famílias e registra a selecionada em cada mídia para estimar
a interação em até 48 h. O aviso usa o corte `compat_score.afinidade_alta` de
`config/pipeline.yaml`. O baseline acadêmico CC3 segue acessível após
`python -m opinion.modelo.treinar`, com `python -m opinion.checkpoint2 --baseline`.
O experimento atual usa `opinion-cc4-aviso-categoria-<midia>` no MLflow.
Os números da EDA e da comparação geral abaixo estão identificados como históricos.
[Regra e resultados por mídia](docs/match_por_midia.md) · [Contrato da pulseira](docs/contrato_mqtt.md)

> **A preencher pela equipe** (exigido pelo item 6 da especificação):
> - **Nome da solução:** _a definir_
> - **Membros:** _Nome Sobrenome (@usuario-github), …_
> - **Google Sites do projeto:** _link_

**Pergunta do modelo (seção 3.4):** dado que dois usuários acabaram de se encontrar, qual a chance de eles
interagirem no OpiniOn nas próximas 48 horas?

Este repositório acumula as entregas de CC sprint a sprint.

| Sprint | Requisito | O que foi entregue | Onde |
|---|---|---|---|
| 1 | **CC1** | Pulseira virtual publicando via MQTT, serviço de ingestão, banco, dataset simulado de 12 semanas, Checkpoint 1 | `src/opinion/` · `docs/contrato_mqtt.md` |
| 2 | **CC2** | Pipeline no Airflow (ingestão, compatibilidade, features, label, preparação), notebook de EDA, validação da simulação, material da sessão com o Design | `src/opinion/pipeline/` · `dags/` · `notebooks/` · `docs/validacao_simulacao.md` |
| 3 | **CC3** | Usuários/obras/avaliações da Amazon Reviews 2023; baseline (Regressão Logística) no MLflow; Checkpoint 2 | `src/opinion/fontes/` · `src/opinion/modelo/` · `docs/fontes_dados.md` |
| 4 | **CC4** | Comparação de 4 modelos no MLflow (matriz de confusão, ROC/AUC, precisão × recall) e análise crítica | `src/opinion/modelo/comparar.py` · `docs/comparacao_modelos.md` |

Usuários, obras e avaliações são **reais** (Amazon Reviews 2023). Encontros da pulseira e o label
ainda são simulados. Qualquer métrica mede essa mistura, não usuários reais da pulseira.

**Chegando agora?** Leia [`docs/contexto.md`](docs/contexto.md) (o projeto em 10 minutos) e
[`docs/status.md`](docs/status.md) (o que está feito e o que falta). Agentes de IA: [`AGENTS.md`](AGENTS.md).

---

## Visão geral

```
 Amazon Reviews 2023 (real)          SPRINT 1 (CC1)                         SPRINT 2 (CC2)              SPRINT 3 (CC3/CC4)
 fontes.amazon2023 ─► data/reais/ ─► gerar_bootstrap ─► pulseira_virtual ─► MQTT ─► ingestao ─► BANCO
                                          │                                      Airflow: opinion_pipeline
                                          └─► carregar_historico_app              … ─► dataset_modelo.parquet
                                                                                              │
                                                                                              ▼
                                                                                    opinion.modelo.comparar --midia ─► MLflow
                                                                                    Checkpoint 2
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

`src/opinion/modelo/comparar.py` compara Regressão logística, Árvore de decisão, Random Forest e
Gradient Boosting no mesmo protocolo do CC3 (escolha e limiar na validação, teste medido uma vez).
Para cada mídia, um experimento próprio no MLflow (`opinion-cc4-aviso-categoria-<midia>`).
Cada modelo tem um run com métricas, matriz de confusão,
curvas ROC e precisão × recall, calibração, importância por permutação e o modelo; no run pai, os
gráficos lado a lado e o intervalo de confiança das diferenças. A comparação geral histórica está em
[`docs/comparacao_modelos.md`](docs/comparacao_modelos.md). O vencedor de cada mídia vai para
`opinion-cc4-aviso-categoria-<midia>@melhor`; os resultados atuais estão em
[`docs/match_por_midia.md`](docs/match_por_midia.md).

```bash
python -m opinion.modelo.comparar --midia musica  # repita para filme, serie, livro e jogo
```

---

## Como rodar

### Opção A — Docker (tudo junto)

```bash
cp .env.example .env        # uma vez: credenciais e segredo locais (o .env não vai para o Git)
docker compose up --build
```

Sobe o broker, o PostgreSQL, a ingestão e a pulseira (replay de cerca de 2 min, depois ao vivo), além do **Airflow** e do Adminer:

| Serviço | Endereço |
|---|---|
| Airflow (sem login, só desenvolvimento) | http://localhost:8080 → DAG `opinion_pipeline` → **Trigger** |
| Adminer (banco) | http://localhost:8081 (PostgreSQL · servidor `postgres` · usuário, senha e base do seu `.env`) |

```bash
docker compose exec ingestao python -m opinion.checkpoint1         # Checkpoint 1
docker compose exec airflow airflow dags test opinion_pipeline     # pipeline sem abrir a interface
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

# CC4 — comparação de modelos
python -m opinion.modelo.comparar --midia musica
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

### 7. Testes automatizados (~35 s)
```
34 passed
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

- **Testado:**
  - fluxo CC1 completo (SQLite + broker local), Checkpoint 1 e DAG no **Airflow 3.1** em rodada anterior;
  - replay do bootstrap atual pela classe de ingestão, com 145.554 mensagens aceitas e 6.972 avisos com os motivos esperados; o broker MQTT da rodada atual ainda precisa de verificação integrada;
  - testes automatizados;
  - notebook executado do início ao fim em rodada anterior; o HTML está marcado como histórico;
  - validação nos 3 cenários, regerada em 18/11/2026: 15 OK, 1 atenção, 1 falha;
  - modelos por mídia no MLflow e demonstração do **Checkpoint 2**;
- **Não testado:** `docker compose` (com PostgreSQL e o container do Airflow), porque o ambiente onde isso foi escrito não tinha Docker. Rodem uma vez antes de apresentar.

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
│   ├── modelo/            CC3/CC4: baseline, modelos por mídia, pontuação de encontro novo
│   ├── fontes/            CC3: download e filtro da Amazon Reviews 2023
│   ├── gerar_bootstrap.py simulador do mundo (encontros, seguidores, interações, label)
│   ├── ingestao.py        recebe MQTT, grava encontros e envia avisos por categoria
│   └── checkpoint1.py, checkpoint2.py
├── embedded/              simulador do sistema embarcado: a pulseira virtual (MQTT)
├── dags/                  DAG do Apache Airflow (opinion_pipeline)
├── api/                   API de inferência (CC7, ainda não implementada)
├── app/                   dashboard (CC9, ainda não implementado)
├── mlruns/                MLflow: mlflow.db e artefatos (fora do Git, só o README é versionado)
├── reports/               relatório em PDF no template SBC
├── config/                parâmetros em YAML: simulação, fontes, pipeline, modelos
├── docs/                  contexto, decisões, status, dicionário de dados, comparação de modelos
├── tests/                 testes automáticos (pytest)
├── infra/                 Mosquitto e Dockerfile do Airflow
├── Dockerfile             imagem dos serviços Python (ingestão, carga, pulseira)
├── docker-compose.yml     a solução inteira com um comando
├── pyproject.toml         torna o pacote de src/ instalável (pip install -e .)
├── requirements.txt       dependências dos serviços
├── requirements-analise.txt  dependências de análise e treino (inclui o pacote)
└── README.md
```

Os comandos `python -m opinion.…` rodam da raiz depois de `pip install -r requirements-analise.txt`,
que instala o pacote de `src/` em modo editável. A pulseira roda com `python -m embedded.pulseira_virtual`.

## Pendências

Decisões com o Design, pendências de infraestrutura e próximos passos ficam em
[`docs/status.md`](docs/status.md).
