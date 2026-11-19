# embedded/ — simulador do sistema embarcado (pulseira virtual)

A trilha de CC usa uma **pulseira virtual** do Sprint 1 ao SR2 (seção 3.4 do documento do projeto).
Ela faz o papel do firmware: publica via **MQTT** o que uma pulseira física publicaria.

| Tópico | Quando |
|---|---|
| `opinion/pulseira/{id}/proximidade` | quando as pulseiras se detectam |
| `opinion/pulseira/{id}/encontro` | ao fim de um encontro entre duas pulseiras |
| `opinion/pulseira/{id}/interesse` | o usuário toca na pulseira |
| `opinion/pulseira/{id}/status` | o usuário troca o modo (aberto, discreto, pausado) |
| `opinion/pulseira/{id}/feedback` | o backend manda o sinal a emitir |

Os IDs dos usuários nas mensagens são anônimos e mudam todo dia. O formato completo está em
[`docs/contrato_mqtt.md`](../docs/contrato_mqtt.md), e o código compartilhado com o backend
(contrato, validação, tokens) fica em `src/opinion/contrato.py`.

## Como rodar

Da raiz do repositório, com o broker MQTT e a ingestão (`python -m opinion.ingestao`) rodando:

```bash
python -m embedded.pulseira_virtual replay                     # publica as 12 semanas do bootstrap (~2 min)
python -m embedded.pulseira_virtual replay --velocidade 3600   # 1 h simulada por segundo
python -m embedded.pulseira_virtual ao-vivo --por-minuto 30    # encontros novos, agora
```

No Docker Compose, os serviços `pulseira-replay` e `pulseira-ao-vivo` rodam esses comandos. O replay publica uma proximidade no início e um encontro ao fim para cada par. O feedback inclui todas as categorias de afinidade alta.
