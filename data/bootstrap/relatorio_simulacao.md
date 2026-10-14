# Relatório da simulação — bootstrap dataset OpiniOn (CC1)

Semente 42 · 1200 usuários · 12 semanas (06/07/2026 a 27/09/2026) · todos os parâmetros são premissas da equipe (`config/simulacao.yaml`).

> Dados **sintéticos**. Qualquer desempenho medido com eles é desempenho sobre a simulação, não sobre o comportamento real dos usuários.

## Volume

| Item | Quantidade |
|---|---|
| usuarios | 1,200 |
| obras | 600 |
| avaliacoes | 60,787 |
| seguidores | 6,579 |
| interacoes | 43,070 |
| encontros | 55,363 |
| toques_interesse | 6,765 |
| trocas_de_modo | 2,396 |

## Label `conectou` (interação em até 48 h)

- Encontros com label positivo: **13.7%** (problema desbalanceado, como esperado)
- Encontros que de fato *causaram* conexão (variável oculta): 5.8%
- Conexões causadas cuja 1ª interação veio depois de 48 h (label = 0 mesmo assim): 3.5%
- Positivos entre amigos (que já se seguem): 27.6% · entre desconhecidos: 8.0%

A diferença entre *conectou* e *causou* é intencional: amigos interagem no app de qualquer jeito, e algumas conexões só acontecem depois da janela. O modelo precisa lidar com esse ruído.

## Por contexto

| Local | Encontros | % conectou | Duração mediana | RSSI médio |
|---|---|---|---|---|
| campus | 19,619 | 14.3% | 240 s | -60.3 dBm |
| cinema | 12,109 | 14.7% | 456 s | -61.0 dBm |
| convencao_games | 12,485 | 15.1% | 442 s | -61.9 dBm |
| livraria | 11,150 | 9.9% | 230 s | -61.8 dBm |

## Por semana (para o split temporal 1–8 / 9–10 / 11–12)

| Semana | Encontros | % conectou |
|---|---|---|
| 1 | 3,837 | 13.4% |
| 2 | 3,876 | 14.3% |
| 3 | 7,836 | 13.8% |
| 4 | 3,738 | 12.8% |
| 5 | 3,570 | 13.4% |
| 6 | 3,525 | 12.4% |
| 7 | 7,702 | 14.5% |
| 8 | 3,560 | 12.6% |
| 9 | 3,522 | 13.8% |
| 10 | 3,208 | 14.1% |
| 11 | 7,526 | 14.3% |
| 12 | 3,463 | 13.1% |

Picos nas semanas 3, 7 e 11 = fins de semana com convenção de games.

## Por persona de quem publicou

| Persona | Encontros | % conectou | Duração mediana | RSSI médio |
|---|---|---|---|---|
| cinefilo | 12,346 | 13.8% | 339 s | -61.2 dBm |
| estudante_casual | 13,100 | 13.0% | 305 s | -61.1 dBm |
| gamer_social | 12,970 | 15.1% | 362 s | -61.2 dBm |
| leitor_reservado | 11,050 | 11.7% | 274 s | -61.3 dBm |
| maratonista_series | 5,897 | 15.3% | 375 s | -60.9 dBm |

