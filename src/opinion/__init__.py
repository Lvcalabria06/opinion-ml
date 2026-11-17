"""
Pacote do OpiniOn (trilha de Machine Learning).

Ao importar, lê o arquivo .env da raiz do projeto (se existir) para as variáveis de
ambiente — sem sobrescrever o que já estiver definido. Crie o seu com:
    cp .env.example .env
"""
import os
from pathlib import Path

# raiz do repositório: src/opinion/__init__.py -> ../.. (ou OPINION_HOME, como em opinion.mundo)
_ENV = Path(os.getenv("OPINION_HOME") or Path(__file__).resolve().parents[2]) / ".env"
if _ENV.exists():
    for _linha in _ENV.read_text(encoding="utf-8").splitlines():
        _linha = _linha.strip()
        if _linha and not _linha.startswith("#") and "=" in _linha:
            _k, _v = _linha.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
