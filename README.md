# OpiniOn — Ciência da Computação (Machine Learning)

**Pergunta do modelo (seção 3.4):** dado que dois usuários acabaram de se encontrar, qual a chance de eles
interagirem no OpiniOn nas próximas 48 horas?

Este repositório acumula as entregas de CC sprint a sprint.

| Sprint | Requisito | O que foi entregue | Onde |
|---|---|---|---|
| 1 | **CC1** | Pulseira virtual publicando via MQTT, serviço de ingestão, banco, dataset simulado de 12 semanas, Checkpoint 1 | `opinion/` · `docs/contrato_mqtt.md` |
| 2 | **CC2** | Pipeline no Airflow (ingestão, compatibilidade, features, label, preparação), notebook de EDA, validação da simulação, material da sessão com o Design | `opinion/pipeline/` · `airflow/dags/` · `notebooks/` · `docs/validacao_simulacao.md` |

Tudo usa **dados sintéticos** (seção 4.3). Qualquer métrica mede a simulação, não usuários reais.

---

## Visão geral

```
 SPRINT 1 (CC1)                                                SPRINT 2 (CC2)
 ─────────────────────────────────────────────────────────    ───────────────────────────────────────────────────
 gerar_bootstrap ─► data/bootstrap/ ─► carregar_historico_app ─┐
                         │                                     ▼
                         └─► pulseira_virtual ─► MQTT ─► ingestao ─► BANCO ─► Airflow: opinion_pipeline
                                                                               ingestao ─┬─ compatibilidade ─┐
                                                                                         ├─ features ────────┼─ preparacao ─ resumo
                                                                                         └─ label ───────────┘        │
                                                                                                                      ▼
                                                                         data/processed/dataset_modelo.parquet  ◄── CC3 (baseline)
                                                                                         │
                                                                    notebooks/eda_encontros.ipynb · validacao_simulacao
```

---

## Sprint 1 — CC1: dados e caminho até o banco

- **`opinion/gerar_bootstrap.py`:** simula 1.200 usuários em 12 semanas (06/07 a 27/09/2026), indo a cinema, livraria, campus e a uma convenção de games (semanas 3, 7 e 11). Gera 54.586 encontros, 42.846 interações no app, 60.787 avaliações, 6.658 toques de interesse e 2.396 trocas de modo. As premissas ficam em `config/simulacao.yaml`.
- **`opinion/pulseira_virtual.py`:** publica encontros, toques e status no contrato MQTT da seção 4.2, com IDs anônimos que mudam todo dia. Tem dois modos: `replay` (as 12 semanas) e `ao-vivo`.
- **`opinion/ingestao.py`:** recebe as mensagens, valida, traduz os IDs, confere o dono da pulseira e grava no banco.
- **`opinion/checkpoint1.py`:** teste do Checkpoint 1 (encontro publicado → broker → banco → consulta).

## Sprint 2 — CC2: pipeline, EDA e validação

### Pipeline no Airflow (`airflow/dags/opinion_pipeline.py`)

| Task | O que faz | Código |
|---|---|---|
| `ingestao` | Lê do banco (o que o MQTT gravou + histórico do app) e grava uma "foto" em `data/staging/`. Para se faltar dado ou houver inconsistência. | `opinion/pipeline/ingestao.py` |
| `compatibilidade` | Similaridade de gosto entre os dois: gêneros, notas nas mesmas obras, favoritas em comum. **Só com avaliações anteriores ao encontro.** | `opinion/pipeline/compatibilidade.py` |
| `features` | Encontro (duração, RSSI, encontros anteriores do par), contexto (local, dia, hora, modo), perfil (atividade, conexões) e relação (já se seguem, já interagiram). **Tudo até o instante do encontro.** | `opinion/pipeline/features.py` |
| `label` | `conectou = 1` se houve interação entre os dois em até 48 h após o fim. Encontros com menos de 48 h ficam "aguardando". | `opinion/pipeline/label.py` |
| `preparacao` | Split temporal (semanas 1–8 / 9–10 / 11–12), purga da fronteira, imputação, one-hot e checagens de qualidade. Grava o dataset do modelo. | `opinion/pipeline/preparacao.py` |
| `resumo` | Registra no log o tamanho e a taxa de conexão de cada split. | — |

As tasks `compatibilidade`, `features` e `label` rodam em paralelo. A DAG roda diariamente e também pode ser disparada na mão.

**Saída** (`data/processed/`), que é a entrada do CC3:

| Split | Encontros | % conectou | Pares |
|---|---|---|---|
| treino (semanas 1–8) | 36.930 | 13,5% | 26.254 |
| validação (9–10) | 6.623 | 13,7% | 4.946 |
| teste (11–12) | 10.501 | 14,2% | 8.152 |

- `dataset_modelo.parquet`: 33 features, `conectou`, `split` e `par_id` (para o k-fold agrupado por par do CC6).
- `features_encontros.parquet`: todos os encontros, inclusive os que ainda aguardam as 48 h. Serve para pontuar encontros novos no CC7.
- `metadados.json`: lista de features por grupo, tamanhos, taxas e corte dos dados.

**Garantias testadas** (`tests/test_pipeline.py`, rode com `python -m pytest tests -q`):
- **Sem informação do futuro:** para 30 encontros sorteados, as features são recalculadas apagando tudo que aconteceu depois do início. O resultado tem de ser idêntico. Também conferimos que o teste falha quando um vazamento é inserido de propósito.
- **Label correto:** 100% de concordância com o gabarito da simulação, e a janela de 48 h está testada nas bordas.
- **Purga:** os 532 encontros do fim da semana 8, cujo label só seria conhecido na semana 9, saem do treino.

A padronização dos dados fica de propósito para o pipeline do modelo (CC3), ajustada só no treino de cada fold, para não vazar informação.

### Notebook de EDA (`notebooks/eda_encontros.ipynb`, já executado; versão HTML ao lado)

Principais achados (entre **desconhecidos**, o público real da pulseira):

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
2. **Checagens de sanidade:** 16 OK e 1 de atenção (RSSI). Uma delas encontrou um **bug real no simulador**, já corrigido: 3 encontros eram registrados com a pulseira pausada.
3. **Sensibilidade:** cenários pessimista, base e otimista (`config/cenarios_sensibilidade.yaml`), rodando o mesmo pipeline e os mesmos modelos.
   - **Se mantém:** a ordem dos contextos, e o fato de o modelo superar a regra (inclusive entre desconhecidos).
   - **Não se mantém:** qual algoritmo é o melhor e quais features mais pesam. Esses resultados não devem ser apresentados como descoberta.
4. **Plano de comparação com dados reais** dos testes do Design (semana 16).

### Sessão com o Design

Os achados para a sessão estão nas seções 3 a 8 e 10 do notebook (e numa apresentação de 10 slides compartilhada com a equipe). A sessão deve fechar estas decisões:
- se a pulseira deve sinalizar quem já se segue;
- a duração mínima antes do sinal;
- os limites por contexto;
- as personas do simulador;
- o custo de um sinal errado.

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

```bash
pip install -r requirements-analise.txt
cp .env.example .env                             # configuração local (uma vez)
# terminal 1: broker MQTT
docker run -p 1883:1883 eclipse-mosquitto:2 mosquitto -c /mosquitto-no-auth.conf

# CC1 — dados chegando pelo MQTT
python -m opinion.carregar_historico_app
python -m opinion.ingestao                       # terminal 2 (fica aberto)
python -m opinion.pulseira_virtual replay        # terminal 3 (~2 min)
python -m opinion.verificar_replay
python -m opinion.checkpoint1

# CC2 — pipeline (escolha um)
python -m opinion.pipeline.executar              # mesmas etapas da DAG, sem Airflow
airflow dags test opinion_pipeline               # com Airflow instalado (AIRFLOW__CORE__DAGS_FOLDER=airflow/dags)

python -m pytest tests -q                        # testes (inclui o de vazamento)
jupyter notebook notebooks/eda_encontros.ipynb   # EDA
python -m opinion.validacao_simulacao            # regera docs/validacao_simulacao.md
```

Sem broker nem banco, o pipeline também lê direto dos arquivos: `python -m opinion.pipeline.executar --fonte arquivos`.

---

## Como testar (e o que esperar)

Siga a **Opção B** acima, nesta ordem. Cada teste prova uma parte dos requisitos.

| # | Comando | Prova | Requisito |
|---|---|---|---|
| 1 | abrir `data/bootstrap/relatorio_simulacao.md` | o dataset de 12 semanas existe e é plausível | CC1 |
| 2 | `python -m opinion.pulseira_virtual replay` | a pulseira publica via MQTT e a ingestão grava | CC1 |
| 3 | `python -m opinion.verificar_replay` | nenhuma mensagem perdida; IDs anônimos resolvidos certo | CC1 |
| 4 | `python -m opinion.checkpoint1` | **Checkpoint 1**: encontro → broker → banco → consulta | CC1 |
| 5 | `python -m opinion.pulseira_virtual ao-vivo --duracao-min 1 --duplicar 0.05 --invalidas 0.05` e depois `python -m opinion.consultar` | a ingestão ignora repetidas e rejeita defeituosas, com o motivo | CC1 |
| 6 | `python -m opinion.pipeline.executar` (ou a DAG no Airflow) | o pipeline gera o dataset do modelo | CC2 |
| 7 | `python -m pytest tests -q` | sem informação do futuro; label e janela de 48 h corretos | CC2 |
| 8 | abrir `notebooks/eda_encontros.html` | EDA e preparação dos dados com justificativas | CC2 |
| 9 | `python -m opinion.validacao_simulacao` | premissas, sanidade e sensibilidade | CC2 |

### 1. Dataset simulado
Em `data/bootstrap/relatorio_simulacao.md`, confira: 1.200 usuários, cerca de 54,6 mil encontros e **cerca de 14% com `conectou = 1`**, além de picos de encontros nas semanas 3, 7 e 11 (convenção).

### 2. Replay pela pulseira virtual (~2 min)
```
[pulseira] Fim do replay: {'encontro': 54586, 'interesse': 6658, 'status': 2396}
```
No terminal da ingestão aparece o progresso a cada 5 mil mensagens, com `'rejeitadas': 0`.

### 3. Nada se perdeu
```
Arquivo do bootstrap  x  banco
  encontros    54,586   54,586   OK
  interesses    6,658    6,658   OK
  status        2,396    2,396   OK

IDs rotativos traduzidos corretamente: 54,586 de 54,586 encontros
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
[compatibilidade]   ...  {'encontros': 54637, ...}
[features       ]   ...
[label          ]   ...  {'rotulados': 54586, 'pendentes': 51, 'taxa_conexao': 0.1364, ...}
[preparacao     ]   ...  {'dataset_modelo': 54054, 'encontros_purgados_fronteira': 532, 'concordancia_label_referencia': 1.0}

Dataset do modelo: {"teste": {"encontros": 10501, ...}, "treino": {"encontros": 36930, ...}, "validacao": {"encontros": 6623, ...}}
```
Os números de encontros e pendentes variam um pouco conforme quantos encontros "ao vivo" você gerou no teste 5. Esses encontros ficam **pendentes** porque ainda não se passaram 48 h. `concordancia_label_referencia: 1.0` quer dizer que o label bate 100% com o gabarito da simulação.

**No Airflow:** com `airflow dags test opinion_pipeline`, ou o botão Trigger na interface, as 6 tasks terminam em `success`, e o log da task `resumo` mostra:
```
Dataset do modelo: 54,054 encontros | taxa de conexão 13.7%
  treino      36,930 encontros  13.5% conectou  26,254 pares
  ...
```

### 7. Testes automatizados (~35 s)
```
4 passed
```
O teste principal recalcula as features de 30 encontros **apagando tudo o que aconteceu depois do início de cada um**. Se algum valor mudar, alguma feature estaria olhando para o futuro, e o teste falha.

### 8. Notebook
Abra `notebooks/eda_encontros.html` no navegador; não precisa instalar nada. Para reexecutar, rode o teste 6 antes e depois `jupyter notebook notebooks/eda_encontros.ipynb` → *Run All*.

### 9. Validação da simulação (~1 min)
No final aparecem 16 checagens `OK`, 1 `ATENÇÃO` (RSSI) e a lista de conclusões que se mantêm ou não nos três cenários. O documento `docs/validacao_simulacao.md` é regerado.

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
  - fluxo CC1 completo (SQLite + broker local): replay sem perdas, 100% dos IDs certos e Checkpoint 1;
  - DAG executada no **Airflow 3.1** lendo do banco preenchido pelo MQTT, com as 6 tasks em sucesso e os encontros ao vivo corretamente "aguardando 48 h";
  - testes automatizados;
  - notebook executado do início ao fim;
  - validação nos 3 cenários.
- **Não testado:** `docker compose` (com PostgreSQL e o container do Airflow), porque o ambiente onde isso foi escrito não tinha Docker. Rodem uma vez antes de apresentar.

## Estrutura

```
config/simulacao.yaml               premissas do simulador
config/pipeline.yaml                label, split temporal, janelas e checagens do pipeline
config/cenarios_sensibilidade.yaml  cenários pessimista / base / otimista
opinion/                            CC1: simulador, pulseira virtual, ingestão MQTT, banco, checkpoint
opinion/pipeline/                   CC2: etapas do pipeline (usadas pela DAG)
opinion/validacao_simulacao.py      CC2: gera docs/validacao_simulacao.md
airflow/dags/opinion_pipeline.py    CC2: a DAG
airflow/Dockerfile                  imagem do Airflow com as dependências
notebooks/eda_encontros.ipynb/.html CC2: EDA executada
tests/test_pipeline.py              vazamento, label, janela de 48 h
docs/                               contrato MQTT, dicionário de dados, validação da simulação
data/bootstrap/                     dataset simulado (CC1)
```

## Pendências com o Design

- Personas definitivas para substituir as provisórias.
- Respostas da tabela de decisões da sessão. As duas primeiras afetam a política de sinal e talvez o label.
- Padrões do tópico `feedback`, e o protótipo físico publicando no mesmo contrato.
