# Reorganização das pastas (item 6) — Plano de execução

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deixar o repositório com a estrutura do item 6 da especificação (`src/`, `embedded/`, `dags/`, `api/`, `app/`, `mlruns/`, `reports/`) sem mudar nenhum resultado.

**Architecture:** Movimentos com `git mv`; o pacote `opinion` vai para `src/opinion/` e é instalável por `pyproject.toml`; a pulseira virtual vira o pacote `embedded`; caminhos derivados (raiz do projeto, DAG, Docker, MLflow, docs) são ajustados. Testes estruturais novos (`tests/test_estrutura.py`) travam a estrutura e a ausência de caminhos antigos.

**Tech Stack:** Python 3.12/3.14, setuptools (pyproject), pytest, MLflow 3, Docker Compose, Airflow 3.1.

**Spec:** `docs/superpowers/specs/2026-11-17-reorganizacao-pastas-design.md`

## Global Constraints
- Movimentos sempre com `git mv` (histórico preservado).
- Nenhuma lógica muda: mesmos números no fim (Task 7).
- Nome do pacote continua `opinion`; comandos `python -m opinion.<modulo>` continuam válidos.
- Pulseira: `python -m embedded.pulseira_virtual …` (rodado da raiz).
- MLflow: `tracking_uri: sqlite:///mlruns/mlflow.db`.
- Raiz do projeto: `OPINION_HOME` se definido; senão `Path(__file__).resolve().parents[2]` de `src/opinion/mundo.py`.
- Código, docs e commits em português; commits no formato `Estrutura: <resumo>`.
- Trabalho num worktree (branch `estrutura-item6`). **Não rodar `pip install -e .` no `.venv` da `main` durante a execução** (apontaria o pacote para o worktree): no worktree, use `PYTHONPATH=src` nos comandos `python -m` e o `pythonpath` do pytest. A instalação editável é feita na `main` depois do merge.
- `data/staging/`, `data/processed/` e `mlflow.db` não existem no worktree (fora do Git): gere com o pipeline quando a task pedir.

## Review Focus
1. Comando rodado fora da raiz (ex.: `jupyter` em `notebooks/`) precisa achar `data/` e `config/` → a raiz vem de `mundo.RAIZ`, não da pasta atual (teste na Task 1).
2. Docker sem o pacote instalado: `RAIZ` precisa apontar para `/app` via `OPINION_HOME` → teste de override na Task 1 e verificação de build na Task 5.
3. Caminho antigo esquecido em docs, compose ou código (`opinion/…`, `airflow/dags`, `opinion.pulseira_virtual`, `sqlite:///mlflow.db`) → teste de varredura na Task 6.
4. Comando do docker-compose apontando para módulo que não existe mais → teste da Task 5.
5. Runs antigos do MLflow sumindo depois da troca do `tracking_uri` → passo de migração do `mlflow.db` e o `checkpoint2` na Task 7.

---

### Task 0: Worktree e números de referência

**Files:** nenhum.

- [ ] **Step 1: Registrar os números de antes, na `main`**

Run (na `main`, com `.venv`): `python -m pytest tests -q`, `python -m opinion.pipeline.executar --fonte arquivos`, `python -m opinion.modelo.treinar --sem-registro`
Expected: `17 passed`; `dataset_modelo: 65443`, `taxa_conexao: 0.1231`; `PR-AUC desconhecidos … 0.180` (teste). Anote as saídas.

- [ ] **Step 2: Criar o worktree** com a skill `superpowers:using-git-worktrees`, branch `estrutura-item6` a partir da `main`.

- [ ] **Step 3: Commitar spec e plano no worktree**

```bash
git add docs/superpowers
git commit -m "Estrutura: design e plano da reorganização (item 6)"
```

---

### Task 1: Pacote em `src/opinion/` + `pyproject.toml`

**Files:**
- Move: `opinion/` → `src/opinion/` (inteiro, por enquanto inclusive `pulseira_virtual.py`)
- Create: `pyproject.toml`
- Modify: `src/opinion/mundo.py` (cálculo de `RAIZ`), `requirements-analise.txt` (+ linha `-e .`)
- Test: `tests/test_estrutura.py` (novo)

**Interfaces:**
- Produces: `opinion.mundo.RAIZ: Path` (raiz do repositório); pacote importável com `pythonpath = ["src", "."]` no pytest.

- [ ] **Step 1: Escrever os testes que falham**

```python
# tests/test_estrutura.py
RAIZ_REPO = Path(__file__).resolve().parents[1]

def test_pacote_mora_em_src():
    import opinion
    assert Path(opinion.__file__).resolve().parent == RAIZ_REPO / "src" / "opinion"

def test_raiz_do_projeto_e_a_raiz_do_repositorio():
    from opinion.mundo import RAIZ, PASTA_BOOTSTRAP
    assert RAIZ == RAIZ_REPO
    assert (PASTA_BOOTSTRAP / "historico_app").is_dir()

def test_opinion_home_sobrescreve_a_raiz(tmp_path):
    # subprocesso com OPINION_HOME=tmp_path e PYTHONPATH=src: imprime opinion.mundo.RAIZ
    assert saida.strip() == str(tmp_path)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest tests/test_estrutura.py -q`
Expected: FAIL (`opinion` ainda em `opinion/`).

- [ ] **Step 3: Mover e configurar**
- `git mv opinion src/opinion`
- `pyproject.toml`: `[project] name = "opinion"`, `version = "0.4.0"`, `requires-python = ">=3.12"`; `[tool.setuptools.packages.find] where = ["src"]`; `[tool.pytest.ini_options] pythonpath = ["src", "."]`, `testpaths = ["tests"]`. As dependências continuam nos `requirements*.txt` (não duplicar no pyproject).
- `mundo.py`: `RAIZ = Path(os.getenv("OPINION_HOME") or Path(__file__).resolve().parents[2])`.
- `requirements-analise.txt`: acrescentar `-e .`.

- [ ] **Step 4: Rodar tudo**

Run: `python -m pytest tests -q`
Expected: `20 passed` (17 antigos + 3 novos).

- [ ] **Step 5: Commit** — `Estrutura: pacote opinion em src/ com pyproject`

---

### Task 2: Pulseira virtual em `embedded/`

**Files:**
- Move: `src/opinion/pulseira_virtual.py` → `embedded/pulseira_virtual.py`
- Create: `embedded/__init__.py` (docstring: simulador do sistema embarcado), `embedded/README.md` (o que é, como rodar, contrato MQTT)
- Modify: docstring de uso dentro de `embedded/pulseira_virtual.py` (`python -m embedded.pulseira_virtual …`), `src/opinion/__init__.py` se listar módulos
- Test: `tests/test_estrutura.py`

**Interfaces:**
- Consumes: `opinion.contrato`, `opinion.mundo` (Task 1).
- Produces: módulo `embedded.pulseira_virtual` com a mesma CLI (`replay`, `ao-vivo`).

- [ ] **Step 1: Testes que falham**

```python
def test_pulseira_mora_em_embedded():
    import embedded.pulseira_virtual as p
    assert Path(p.__file__).resolve().parent == RAIZ_REPO / "embedded"
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("opinion.pulseira_virtual")

def test_cli_da_pulseira_carrega():
    # subprocesso: python -m embedded.pulseira_virtual --help, cwd=RAIZ_REPO, PYTHONPATH=src
    assert r.returncode == 0 and "replay" in r.stdout
```

- [ ] **Step 2: Rodar e ver falhar** — `python -m pytest tests/test_estrutura.py -q` → FAIL em `embedded`.
- [ ] **Step 3: `git mv` + `__init__.py` + `README.md`**
- [ ] **Step 4: Rodar tudo** — `python -m pytest tests -q` → `22 passed`.
- [ ] **Step 5: Commit** — `Estrutura: pulseira virtual em embedded/`

---

### Task 3: DAG em `dags/` e Dockerfile do Airflow em `infra/airflow/`

**Files:**
- Move: `airflow/dags/opinion_pipeline.py` → `dags/opinion_pipeline.py`; `airflow/Dockerfile` → `infra/airflow/Dockerfile` (a pasta `airflow/` deixa de existir)
- Modify: `dags/opinion_pipeline.py` (raiz = `OPINION_HOME` ou `parents[1]`; insere `<raiz>/src` no `sys.path`)
- Test: `tests/test_estrutura.py`

**Interfaces:**
- Produces: `dags/opinion_pipeline.py` com `RAIZ_PROJETO: Path` e a DAG `opinion_pipeline` (nomes das tasks inalterados).

- [ ] **Step 1: Teste que falha**

```python
def test_dag_acha_o_pacote(monkeypatch):
    # injeta em sys.modules um módulo falso "airflow.sdk" com dag/task que só devolvem a função
    # carrega dags/opinion_pipeline.py com importlib.util.spec_from_file_location
    assert mod.RAIZ_PROJETO == RAIZ_REPO
    assert str(RAIZ_REPO / "src") in sys.path
```

- [ ] **Step 2: Rodar e ver falhar** — arquivo não existe em `dags/`.
- [ ] **Step 3: `git mv` dos dois arquivos e ajuste da raiz/`sys.path` na DAG**
- [ ] **Step 4: Rodar tudo** — `python -m pytest tests -q` → `23 passed`.
- [ ] **Step 5: Commit** — `Estrutura: DAG em dags/ e Dockerfile do Airflow em infra/`

---

### Task 4: MLflow em `mlruns/`

**Files:**
- Modify: `config/modelo.yaml` (`tracking_uri: sqlite:///mlruns/mlflow.db`), `.gitignore` (trocar `mlruns/` e `mlflow.db` por `mlruns/*` + `!mlruns/README.md`)
- Create: `mlruns/README.md` (o que fica ali, por que não vai para o Git, comando do `mlflow ui`)
- Test: `tests/test_estrutura.py`

- [ ] **Step 1: Teste que falha**

```python
def test_mlflow_fica_em_mlruns():
    from opinion.modelo.treinar import config
    assert config()["mlflow"]["tracking_uri"] == "sqlite:///mlruns/mlflow.db"
    assert (RAIZ_REPO / "mlruns" / "README.md").is_file()
```

- [ ] **Step 2: Rodar e ver falhar**
- [ ] **Step 3: Alterar config, `.gitignore` e criar o README**; conferir com `git check-ignore -v mlruns/x.db` (ignorado) e `git check-ignore mlruns/README.md` (não ignorado).
- [ ] **Step 4: Verificar no worktree**

Run: `PYTHONPATH=src python -m opinion.pipeline.executar --fonte arquivos` e depois `PYTHONPATH=src python -m opinion.modelo.treinar`
Expected: cria `mlruns/mlflow.db`; `opinion-baseline-cc3 v1 @champion`; PR-AUC desconhecidos 0.180.

- [ ] **Step 5: Rodar tudo e commit** — `24 passed`; `Estrutura: MLflow (mlflow.db e artefatos) em mlruns/`

---

### Task 5: Docker (Dockerfile + docker-compose)

**Files:**
- Modify: `Dockerfile` (COPY `src`, `embedded`, `config`; `ENV PYTHONPATH=/app/src OPINION_HOME=/app`), `docker-compose.yml` (comandos da pulseira → `python -m embedded.pulseira_virtual …`; serviço `airflow`: `dockerfile: infra/airflow/Dockerfile`, volumes `./src:/opt/opinion/src:ro`, `./dags:/opt/opinion/dags:ro`, `PYTHONPATH=/opt/opinion/src`, `AIRFLOW__CORE__DAGS_FOLDER=/opt/opinion/dags`; comentários do topo atualizados)
- Test: `tests/test_estrutura.py`

- [ ] **Step 1: Teste que falha**

```python
def test_compose_so_aponta_para_o_que_existe():
    c = yaml.safe_load((RAIZ_REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    # todo "python -m X" em command: X resolve para src/X.py, src/X/__init__.py ou ./X.py (embedded)
    # todo volume "./algo:..." existe; todo build.dockerfile existe
    assert faltando == []
```

- [ ] **Step 2: Rodar e ver falhar** (comandos `opinion.pulseira_virtual` e volumes `./opinion`, `./airflow/dags`).
- [ ] **Step 3: Ajustar `Dockerfile` e `docker-compose.yml`**
- [ ] **Step 4: Verificar**

Run: `python -m pytest tests -q` → `25 passed`; `docker compose config --quiet` → sem erro; `docker compose build` → todas as imagens construídas (rodar em segundo plano; pode levar minutos).

- [ ] **Step 5: Commit** — `Estrutura: Docker com src/, embedded/ e dags/`

---

### Task 6: Pastas `api/`, `app/`, `reports/` e documentação

**Files:**
- Create: `api/README.md` (API de inferência — CC7; base: `src/opinion/modelo/pontuar.py`), `app/README.md` (dashboard — CC9; indicadores da seção 3.2), `reports/README.md` (PDF no template SBC, obrigatório)
- Modify: `README.md` (nova seção "Estrutura do repositório" mapeando cada pasta ao item 6; seção "Estrutura" antiga; todos os comandos/caminhos), `AGENTS.md`, `docs/contexto.md`, `docs/status.md`, `docs/contrato_mqtt.md`, `docs/fontes_dados.md`, `docs/decisoes.md` (D-012: reorganização conforme o item 6, abordagem A)
- Test: `tests/test_estrutura.py`

- [ ] **Step 1: Testes que falham**

```python
PASTAS_ITEM6 = ["data", "notebooks", "src", "embedded", "dags", "api", "app", "mlruns", "reports"]
ARQUIVOS_ITEM6 = ["Dockerfile", "docker-compose.yml", "requirements.txt", "README.md"]

def test_estrutura_do_item_6():
    assert [p for p in PASTAS_ITEM6 if not (RAIZ_REPO / p).is_dir()] == []
    assert [a for a in ARQUIVOS_ITEM6 if not (RAIZ_REPO / a).is_file()] == []

ANTIGOS = ["opinion/", "airflow/dags", "airflow/Dockerfile", "opinion.pulseira_virtual", "sqlite:///mlflow.db"]

def test_sem_caminhos_antigos():
    # varre *.md (fora de docs/superpowers/), *.yml, *.yaml, *.py, *.toml, Dockerfile*
    # "opinion/" só conta se NÃO for precedido de "src/" (regex (?<!src/)\bopinion/)
    assert sobras == []
```

- [ ] **Step 2: Rodar e ver falhar**
- [ ] **Step 3: Criar os READMEs e atualizar a documentação** até o teste passar. O link do Google Sites e os membros do README ficam como pendência anotada (não inventar).
- [ ] **Step 4: Rodar tudo** — `python -m pytest tests -q` → `27 passed`.
- [ ] **Step 5: Commit** — `Estrutura: api/, app/, reports/ e documentação atualizada`

---

### Task 7: Verificação final contra os números de referência

**Files:** nenhum (só verificação).

- [ ] **Step 1: Suíte completa** — `python -m pytest tests -q` → `27 passed`.
- [ ] **Step 2: Mesmos números da Task 0** (no worktree, `PYTHONPATH=src`):
  `python -m opinion.pipeline.executar --fonte arquivos` → `65443`, `0.1231`;
  `python -m opinion.modelo.treinar --sem-registro` → PR-AUC desconhecidos `0.180`;
  `python -m opinion.modelo.comparar --sem-registro` → Gradient Boosting escolhido, `0.197`;
  `python -m opinion.checkpoint2` → `CHECKPOINT 2 OK`;
  `python -m opinion.ingestao --help` e `python -m embedded.pulseira_virtual --help` → carregam sem erro.
- [ ] **Step 3: Revisão de código** com `superpowers:requesting-code-review` contra o spec.
- [ ] **Step 4: Finalizar** com `superpowers:finishing-a-development-branch` (perguntar: merge, PR, manter ou descartar). Depois do merge, na `main`: `pip install -r requirements-analise.txt` (instala o pacote editável) e mover o `mlflow.db` local para `mlruns/` (`mv mlflow.db mlruns/`), conferindo que `python -m opinion.checkpoint2` carrega o `@champion` antigo.
