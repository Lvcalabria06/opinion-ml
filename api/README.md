# api/ — API de inferência (CC7)

FastAPI em `api/main.py`. Recebe um encontro e devolve a chance de os dois usuários interagirem no
OpiniON em até 48 h, usando o modelo de produção do MLflow (`opinion-producao@producao`; na falta
dele, `opinion-cc5-otimizado@otimizado` ou `opinion-cc4-aviso-categoria@melhor`).

```bash
uvicorn api.main:app --port 8000          # da raiz; documentação interativa em http://localhost:8000/docs
```

| Rota | O que faz |
|---|---|
| `GET /saude` | estado da API, do banco e do modelo |
| `GET /modelo` | modelo servido, limiar, calibração e métricas de validação/teste |
| `POST /prever` | encontro no formato do contrato MQTT (IDs anônimos do dia) → chance, nível do sinal, afinidade por mídia |
| `POST /prever/usuarios` | o mesmo com IDs reais (`usr-…`), para demonstração |
| `GET /inferencias` | últimas inferências gravadas (consumidor MQTT e API) |
| `GET /encontros/recentes` | últimos encontros gravados pela ingestão, com o aviso de afinidade |
| `GET /indicadores` | taxa de conversão, precisão dos sinais e taxa de silenciamento |

Exemplo:

```bash
curl -X POST localhost:8000/prever/usuarios -H 'content-type: application/json' \
     -d '{"usuario_a": "usr-00694", "usuario_b": "usr-01029", "local": "cinema", "duracao_s": 600}'
```

As features são calculadas por `opinion.inferencia.Inferencia`, com as mesmas funções do pipeline do
CC2 e só com dados anteriores ao início do encontro. Os dados (banco) e o modelo são recarregados a
cada `OPINION_RECARREGAR_S` segundos (padrão 120): um modelo novo promovido pela DAG entra sozinho.
O consumo automático dos encontros da pulseira fica em `src/opinion/inferencia_mqtt.py`.
