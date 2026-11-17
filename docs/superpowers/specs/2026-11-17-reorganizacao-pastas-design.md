# Reorganização das pastas conforme a especificação (item 6)

## Intenção
Deixar o repositório com a estrutura sugerida no item 6 da Especificação de Projeto (CESAR School,
Aprendizado de Máquina 2026.6), para que quem corrige encontre cada parte da solução onde espera:
`data/`, `notebooks/`, `src/`, `embedded/`, `dags/`, `api/`, `app/`, `mlruns/`, `reports/`,
`Dockerfile`, `docker-compose.yml`, `requirements.txt`, `README.md`.

## Objetivos
- Mesma estrutura do item 6, com código de verdade em `src/`, `embedded/` e `dags/`.
- Todo comando que funciona hoje continua funcionando depois (com o nome novo, quando mudar).
- Resultados idênticos: mesmo dataset, mesmas métricas, testes passando.

## Fora do escopo
- Escrever a API (CC7), o dashboard (CC9) ou o relatório SBC: só as pastas, com um README dizendo o que vai nelas.
- Renomear o pacote `opinion` ou mudar qualquer lógica.
- Testar `docker compose up` de ponta a ponta (já é uma pendência à parte).

## Anti-objetivos (seria fracasso mesmo "funcionando")
- Qualquer número do modelo mudar.
- Um comando do README, do AGENTS.md ou do docker-compose parar de funcionar sem estar atualizado.
- Perder o histórico do Git dos arquivos movidos.

## Abordagens consideradas
- **A — separar de verdade (escolhida pelo Lucas):** pulseira virtual em `embedded/`, o resto do
  pacote em `src/opinion/`, `pyproject.toml` para instalar o pacote e manter `python -m opinion...`.
- B — pacote inteiro em `src/`, `embedded/` só com um atalho: quebra menos, mas a pasta fica de fachada.
- C — manter `opinion/` na raiz e só criar as pastas que faltam: menor mudança, mas não segue o item 6
  (sem `src/`).

## Estrutura final
```
/
├── data/                 sem mudança
├── notebooks/            sem mudança
├── src/opinion/          ← opinion/ (tudo, menos pulseira_virtual.py)
├── embedded/             ← opinion/pulseira_virtual.py  (+ README.md)
├── dags/                 ← airflow/dags/opinion_pipeline.py
├── api/README.md         novo: o que entra no CC7
├── app/README.md         novo: o que entra no CC9
├── mlruns/               artefatos do MLflow + mlflow.db (conteúdo fora do Git; README.md versionado)
├── reports/README.md     novo: onde vai o PDF no template SBC
├── config/ docs/ tests/ infra/   mantidas (o item 6 permite ajustes)
├── Dockerfile  docker-compose.yml  requirements.txt  requirements-analise.txt
├── pyproject.toml        novo
└── README.md
```

## Decisões de como fazer (chamadas minhas, para conferir)
1. **Movimentos com `git mv`**, para preservar o histórico.
2. **`embedded/` como pacote** (`embedded/__init__.py`, `embedded/pulseira_virtual.py`), rodado da raiz:
   `python -m embedded.pulseira_virtual replay`. Ele importa `opinion.contrato` e `opinion.mundo`
   (o contrato MQTT é compartilhado com a ingestão do backend, por isso fica em `src/`).
3. **`gerar_bootstrap.py` fica em `src/opinion/`**: ele gera o dataset (o "mundo" simulado); a pulseira
   só reproduz o que ele gerou.
4. **Instalação:** `pyproject.toml` declara o pacote `opinion` em `src/`. O `requirements-analise.txt`
   ganha `-e .`, então o comando de instalação que a equipe já usa
   (`pip install -r requirements-analise.txt`) instala o pacote sem passo novo.
5. **Raiz do projeto:** `opinion/mundo.py` calcula a raiz como `parents[2]` de `src/opinion/mundo.py`,
   com `OPINION_HOME` como alternativa (a DAG e o Docker já usam essa variável).
6. **Testes:** `pyproject.toml` configura o pytest com `pythonpath = ["src", "."]`, para rodar mesmo sem instalar.
7. **MLflow:** `config/modelo.yaml` passa a usar `sqlite:///mlruns/mlflow.db`. O `mlflow.db` local é
   movido para `mlruns/` (os runs existentes continuam válidos, porque os artefatos já estão em `mlruns/`).
   `.gitignore` ignora o conteúdo de `mlruns/`, menos o `README.md`. `mlflow ui` passa a ser
   `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
8. **Airflow:** a DAG vai para `dags/`; ela acha a raiz por `parents[1]` (ou `OPINION_HOME`) e coloca
   `src/` no `sys.path`. `airflow/Dockerfile` vai para `infra/airflow/Dockerfile`; a pasta `airflow/` some.
9. **Docker:** o `Dockerfile` copia `src/`, `embedded/` e `config/` e define `PYTHONPATH=/app/src` e
   `OPINION_HOME=/app` (sem instalar o pacote, para a raiz continuar certa). O docker-compose troca os
   volumes e caminhos (`./src`, `./dags`, `infra/airflow/Dockerfile`) e os comandos da pulseira para
   `python -m embedded.pulseira_virtual`.
10. **Documentação:** README (estrutura, comandos e uma seção nova mapeando cada pasta ao item 6),
    AGENTS.md, `docs/contexto.md`, `docs/status.md`, `docs/contrato_mqtt.md`, `docs/fontes_dados.md`:
    todo caminho `opinion/…`, `airflow/dags` e `opinion.pulseira_virtual` atualizado. Nova decisão em
    `docs/decisoes.md`.

## Verificação
- `pytest tests -q` passando (17 testes).
- `python -m opinion.pipeline.executar --fonte arquivos`: mesmos números (65.443 encontros, 12,31%).
- `python -m opinion.modelo.treinar --sem-registro`: mesma PR-AUC (0,180 entre desconhecidos).
- `python -m opinion.checkpoint2`: CHECKPOINT 2 OK, lendo o MLflow do lugar novo.
- `python -m embedded.pulseira_virtual --help` e `python -m opinion.ingestao --help` carregam.
- `docker compose config` válido e `docker compose build` concluindo.
- Busca por caminhos antigos (`opinion/`, `airflow/dags`, `opinion.pulseira_virtual`) sem sobras.

## Deixado para quem executa
- Texto dos READMEs de `api/`, `app/`, `reports/`, `embedded/` e `mlruns/`.
- Ordem exata dos commits (um por etapa ou um só).
