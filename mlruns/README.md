# mlruns/ — MLflow

Tudo o que o MLflow grava fica aqui:

- `mlflow.db`: banco (SQLite) com experimentos, runs, parâmetros, métricas e o Model Registry;
- `<id do experimento>/…`: artefatos de cada run (modelos, gráficos, `resumo.json`, `explicacao.txt`).

| Experimento | Gerado por | Conteúdo |
|---|---|---|
| `opinion-cc3-aviso-categoria` | `python -m opinion.modelo.treinar` | baseline geral opcional (`opinion-baseline-cc3-aviso-categoria@champion`) |
| `opinion-cc4-aviso-categoria-<midia>` | `python -m opinion.modelo.comparar --midia <midia>` | quatro famílias no recorte da mídia; vencedor em `opinion-cc4-aviso-categoria-<midia>@melhor` |
| `opinion-cc4-aviso-categoria` | `python -m opinion.modelo.comparar` | comparação geral acadêmica, separada dos recortes |

O conteúdo **não vai para o Git** (são arquivos binários e grandes, e cada pessoa gera os seus
rodando os comandos acima). Só este README é versionado. O endereço do banco está em
`config/modelo.yaml` (`tracking_uri: sqlite:///mlruns/mlflow.db`); os comandos rodam da raiz do repositório.

**Já tinha o projeto antes da reorganização (D-012)?** O banco ficava na raiz. Mova-o uma vez, da raiz
do repositório, **antes** de rodar `treinar` ou `comparar` (senão nasce um banco novo vazio e o
`@champion` some):

```bash
mv mlflow.db mlruns/
```

Os runs antigos continuam válidos: os artefatos deles já estavam em `mlruns/`.

Para abrir a interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db     # http://localhost:5000
```
