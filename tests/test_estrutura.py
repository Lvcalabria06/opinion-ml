"""Estrutura do repositório conforme o item 6 da especificação (src/, embedded/, dags/, …)."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ_REPO = Path(__file__).resolve().parents[1]


def _python(codigo_ou_args: list[str], **env) -> subprocess.CompletedProcess:
    """Roda o Python do ambiente atual na raiz do repositório, com src/ no caminho de imports."""
    ambiente = {**os.environ, "PYTHONPATH": str(RAIZ_REPO / "src"), **env}
    return subprocess.run([sys.executable, *codigo_ou_args], cwd=RAIZ_REPO, env=ambiente,
                          capture_output=True, text=True, timeout=120)


def test_pacote_mora_em_src():
    import opinion
    assert Path(opinion.__file__).resolve().parent == RAIZ_REPO / "src" / "opinion"


def test_raiz_do_projeto_e_a_raiz_do_repositorio():
    from opinion.mundo import PASTA_BOOTSTRAP, RAIZ
    assert RAIZ == RAIZ_REPO
    assert (PASTA_BOOTSTRAP / "historico_app").is_dir()


def test_env_da_raiz_e_lido_ao_importar():
    ambiente = {k: v for k, v in os.environ.items() if k != "OPINION_SEGREDO_TOKEN"}
    ambiente["PYTHONPATH"] = str(RAIZ_REPO / "src")
    r = subprocess.run([sys.executable, "-c", "import os, opinion; print(os.environ.get('OPINION_SEGREDO_TOKEN'))"],
                       cwd=RAIZ_REPO, env=ambiente, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() not in ("", "None")


def test_opinion_home_sobrescreve_a_raiz(tmp_path):
    r = _python(["-c", "from opinion.mundo import RAIZ; print(RAIZ)"], OPINION_HOME=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert Path(r.stdout.strip()) == tmp_path


# ------------------------------------------------------------------ embedded/
def test_pulseira_mora_em_embedded():
    import embedded.pulseira_virtual as p
    assert Path(p.__file__).resolve().parent == RAIZ_REPO / "embedded"
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("opinion.pulseira_virtual")


def test_cli_da_pulseira_carrega():
    r = _python(["-m", "embedded.pulseira_virtual", "--help"])
    assert r.returncode == 0, r.stderr
    assert "replay" in r.stdout


# ------------------------------------------------------------------ dags/
def test_dag_acha_o_pacote(monkeypatch):
    """Carrega dags/opinion_pipeline.py sem Airflow instalado (decoradores falsos; a DAG não executa)."""
    import types
    falso = types.ModuleType("airflow.sdk")
    falso.dag = lambda *a, **k: (lambda f: (lambda *x, **y: None))
    falso.task = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    monkeypatch.setitem(sys.modules, "airflow", types.ModuleType("airflow"))
    monkeypatch.setitem(sys.modules, "airflow.sdk", falso)
    monkeypatch.delenv("OPINION_HOME", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    caminho = RAIZ_REPO / "dags" / "opinion_pipeline.py"
    assert caminho.is_file()
    spec = importlib.util.spec_from_file_location("opinion_pipeline_teste", caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.RAIZ_PROJETO == RAIZ_REPO
    assert str(RAIZ_REPO / "src") in sys.path
    # sobras ignoradas pelo git (logs, __pycache__) podem ficar em airflow/; a DAG antiga não
    assert not (RAIZ_REPO / "airflow" / "dags" / "opinion_pipeline.py").exists()
    assert (RAIZ_REPO / "infra" / "airflow" / "Dockerfile").is_file()


# ------------------------------------------------------------------ mlruns/
def test_mlflow_fica_em_mlruns():
    from opinion.modelo.treinar import config
    assert config()["mlflow"]["tracking_uri"] == "sqlite:///mlruns/mlflow.db"
    assert (RAIZ_REPO / "mlruns" / "README.md").is_file()


# ------------------------------------------------------------------ Docker
def _modulo_existe(nome: str) -> bool:
    partes = nome.split(".")
    for base in (RAIZ_REPO / "src", RAIZ_REPO):       # pacote opinion (src/) ou embedded (raiz)
        p = base.joinpath(*partes)
        if p.with_suffix(".py").is_file() or (p / "__init__.py").is_file():
            return True
    return False


def test_compose_so_aponta_para_o_que_existe():
    import re

    import yaml
    c = yaml.safe_load((RAIZ_REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    servicos = dict(c["services"])
    padrao = c.get("x-app", {})
    faltando = []
    for nome, s in servicos.items():
        comando = s.get("command") or ""
        comando = " ".join(comando) if isinstance(comando, list) else comando
        faltando += [f"{nome}: módulo {m}" for m in re.findall(r"python -m ([\w.]+)", comando) if not _modulo_existe(m)]
        build = s.get("build") or (padrao.get("build") if s.get("<<") else None)
        if isinstance(build, dict) and build.get("dockerfile"):
            if not (RAIZ_REPO / build.get("context", ".") / build["dockerfile"]).is_file():
                faltando.append(f"{nome}: dockerfile {build['dockerfile']}")
        for v in (s.get("volumes") or []) + (padrao.get("volumes") or []):
            origem = str(v).split(":")[0]
            if origem.startswith("./") and not (RAIZ_REPO / origem).exists():
                faltando.append(f"{nome}: volume {origem}")
    assert faltando == []


# ------------------------------------------------------------------ item 6 e documentação
PASTAS_ITEM6 = ["data", "notebooks", "src", "embedded", "dags", "api", "app", "mlruns", "reports"]
ARQUIVOS_ITEM6 = ["Dockerfile", "docker-compose.yml", "requirements.txt", "README.md"]


def test_egg_info_do_pip_install_e_ignorado():
    r = subprocess.run(["git", "check-ignore", "-q", "src/opinion.egg-info/PKG-INFO"], cwd=RAIZ_REPO)
    assert r.returncode == 0


def test_migracao_do_mlflow_documentada():
    # quem já tinha o projeto precisa mover o banco antigo, senão o @champion some
    for doc in ("README.md", "mlruns/README.md"):
        assert "mv mlflow.db mlruns/" in (RAIZ_REPO / doc).read_text(encoding="utf-8"), doc


def test_estrutura_do_item_6():
    assert [p for p in PASTAS_ITEM6 if not (RAIZ_REPO / p).is_dir()] == []
    assert [a for a in ARQUIVOS_ITEM6 if not (RAIZ_REPO / a).is_file()] == []
    # pastas que ainda não têm código (CC7, CC9, relatório) explicam o que vai nelas
    assert [p for p in ("api", "app", "reports") if not (RAIZ_REPO / p / "README.md").is_file()] == []


# caminhos de antes da reorganização; "opinion/pulseira/..." é tópico MQTT, não caminho
ANTIGOS = [r"(?<![\w/.-])opinion/(?!pulseira)", r"airflow/dags", r"(?<!infra/)airflow/Dockerfile",
           r"opinion\.pulseira_virtual", r"sqlite:///mlflow\.db"]
FORA_DA_VARREDURA = {".git", ".venv", ".claude", ".superpowers", "data", "mlruns", "notebooks", "__pycache__"}


def test_sem_caminhos_antigos():
    import re
    padrao = re.compile("|".join(ANTIGOS))
    sobras = []
    for arq in RAIZ_REPO.rglob("*"):
        rel = arq.relative_to(RAIZ_REPO)
        if (not arq.is_file() or FORA_DA_VARREDURA & set(rel.parts) or rel.parts[:2] == ("docs", "superpowers")
                or arq.name == "test_estrutura.py"):
            continue
        if not (arq.suffix in {".md", ".yml", ".yaml", ".py", ".toml"} or arq.name.startswith("Dockerfile")):
            continue
        for n, linha in enumerate(arq.read_text(encoding="utf-8").splitlines(), 1):
            if padrao.search(linha):
                sobras.append(f"{rel}:{n}: {linha.strip()[:80]}")
    assert sobras == []
