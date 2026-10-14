"""
Pacote do OpiniOn (trilha de Machine Learning).

Ao importar, lê o arquivo .env da raiz do projeto (se existir) para as variáveis de
ambiente — sem sobrescrever o que já estiver definido. Crie o seu com:
    cp .env.example .env
"""
import os
from pathlib import Path

_ENV = Path(__file__).resolve().parent.parent / ".env"
if _ENV.exists():
    for _linha in _ENV.read_text(encoding="utf-8").splitlines():
        _linha = _linha.strip()
        if _linha and not _linha.startswith("#") and "=" in _linha:
            _k, _v = _linha.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
