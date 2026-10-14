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
| Adminer (banco) | http://localhost:8081 (PostgreSQL · servidor `postgres` · `opinion`/`opinion`) |

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
