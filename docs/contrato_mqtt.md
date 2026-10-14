# Contrato de mensagens MQTT — pulseira OpiniOn (v1)

Implementa a seção 4.2 do documento do projeto. Código: `opinion/contrato.py`.
Artefato compartilhado com o Design: o protótipo físico deve publicar exatamente estas mensagens.

| Tópico | Direção | QoS | Quando |
|---|---|---|---|
| `opinion/pulseira/{id}/encontro` | pulseira → backend | 1 | ao fim de um encontro |
| `opinion/pulseira/{id}/interesse` | pulseira → backend | 1 | usuário toca na pulseira |
| `opinion/pulseira/{id}/status` | pulseira → backend | 1 | usuário troca o modo de descoberta |
| `opinion/pulseira/{id}/feedback` | backend → pulseira | 1 | sinal a emitir (a partir do CC7) |

`{id}` é o identificador da pulseira (`PUL-0001`). O backend sabe de quem é cada pulseira.

## IDs anônimos e rotativos

Os usuários aparecem nas mensagens como `u-` + 10 caracteres, por exemplo `u-3f9a1c2b7e`:

```
token = "u-" + HMAC-SHA256(segredo, "usr-00017|2026-11-03")[:10]
```

O token **muda todo dia**. É isso que uma pulseira anuncia por Bluetooth e o que a outra
pulseira enxerga. Quem capturar as mensagens não descobre quem é a pessoa nem consegue
segui-la de um dia para o outro. Só o backend, que tem o segredo (`OPINION_SEGREDO_TOKEN`),
traduz o token para o usuário.

## `encontro`

```json
{"v": 1, "encontro_id": "e-001042", "usuario_a": "u-3f9a1c2b7e", "usuario_b": "u-8c01d2e4aa",
 "inicio": "2026-11-03T19:42:10", "duracao_s": 340, "rssi_medio": -61.0, "local": "cinema"}
```

| Campo | Regra |
|---|---|
| `v` | versão do contrato, `1` |
| `encontro_id` | único; reenvio com o mesmo id é ignorado |
| `usuario_a` | token do **dono da pulseira que publicou** (o backend confere) |
| `usuario_b` | token da outra pessoa, diferente de `usuario_a` |
| `inicio` | data/hora local ISO-8601 |
| `duracao_s` | número > 0 |
| `rssi_medio` | intensidade média do sinal, entre -110 e 0 dBm |
| `local` | `convencao_games`, `cinema`, `livraria` ou `campus` |

## `interesse`

```json
{"v": 1, "encontro_id": "e-001042", "usuario": "u-3f9a1c2b7e", "ts": "2026-11-03T19:48:02"}
```

O toque é dado secundário: vai para a EDA, mas não é feature nem label do modelo (seção 4.4).

## `status`

```json
{"v": 1, "usuario": "u-3f9a1c2b7e", "modo": "discreto", "ts": "2026-11-03T08:10:00"}
```

`modo`: `aberto`, `discreto` ou `pausado`. Uma pulseira **pausada não anuncia nem registra
encontros**. O histórico de modos alimenta a taxa de silenciamento (seção 3.2).

## `feedback` (proposta inicial — fechar com o Design)

```json
{"v": 1, "encontro_id": "e-001042", "nivel": "periferico_leve", "padrao": "luz_suave", "ts": "2026-11-03T19:48:00"}
```

`nivel`: `nenhum`, `periferico_leve` ou `periferico_perceptivel` (faixas da seção 4.5).
`padrao` é definido pelo Design. No Sprint 1 a pulseira virtual já escuta esse tópico, mas o
backend ainda não envia nada: isso começa no CC7.

## O que o serviço de ingestão faz com cada mensagem

1. Grava a mensagem bruta em `mqtt_mensagens`, inclusive as rejeitadas, com o motivo.
2. Valida os campos acima.
3. Traduz os tokens para usuários. Token desconhecido é rejeitado.
4. Confere se o token do publicador é do dono da pulseira do tópico.
5. Grava em `encontros`, `interesses` ou `status_pulseira`. Reentregas do QoS 1 são ignoradas.
