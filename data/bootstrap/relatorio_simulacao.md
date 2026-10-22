# Relatório da simulação — bootstrap dataset OpiniOn (CC1)

Semente 42 · 1200 usuários · 12 semanas (06/07/2026 a 27/09/2026) · todos os parâmetros são premissas da equipe (`config/simulacao.yaml`).

> Dados **sintéticos**. Qualquer desempenho medido com eles é desempenho sobre a simulação, não sobre o comportamento real dos usuários.

## Volume

| Item | Quantidade |
|---|---|
| usuarios | 1,200 |
| obras | 600 |
| avaliacoes | 60,787 |
| seguidores | 6,541 |
| interacoes | 42,846 |
| encontros | 54,586 |
| toques_interesse | 6,658 |
| trocas_de_modo | 2,396 |

## Label `conectou` (interação em até 48 h)

- Encontros com label positivo: **13.6%** (problema desbalanceado, como esperado)
- Encontros que de fato *causaram* conexão (variável oculta): 5.6%
- Conexões causadas cuja 1ª interação veio depois de 48 h (label = 0 mesmo assim): 3.3%
- Positivos entre amigos (que já se seguem): 27.6% · entre desconhecidos: 7.8%

A diferença entre *conectou* e *causou* é intencional: amigos interagem no app de qualquer jeito, e algumas conexões só acontecem depois da janela. O modelo precisa lidar com esse ruído.

## Por contexto

| Local | Encontros | % conectou | Duração mediana | RSSI médio |
|---|---|---|---|---|
| campus | 19,783 | 14.1% | 239 s | -60.4 dBm |
| cinema | 11,857 | 14.7% | 461 s | -61.1 dBm |
| convencao_games | 12,089 | 14.9% | 446 s | -61.9 dBm |
| livraria | 10,857 | 10.2% | 228 s | -61.8 dBm |

## Por semana (para o split temporal 1–8 / 9–10 / 11–12)

| Semana | Encontros | % conectou |
|---|---|---|
| 1 | 3,837 | 13.6% |
| 2 | 3,876 | 12.7% |
| 3 | 7,836 | 13.9% |
| 4 | 3,738 | 13.4% |
| 5 | 3,570 | 12.9% |
| 6 | 3,525 | 12.6% |
| 7 | 7,665 | 13.9% |
| 8 | 3,415 | 13.9% |
| 9 | 3,372 | 13.6% |
| 10 | 3,251 | 13.8% |
| 11 | 7,081 | 14.5% |
| 12 | 3,420 | 13.5% |

Picos nas semanas 3, 7 e 11 = fins de semana com convenção de games.

## Por persona de quem publicou

| Persona | Encontros | % conectou | Duração mediana | RSSI médio |
|---|---|---|---|---|
| cinefilo | 12,221 | 13.9% | 346 s | -61.2 dBm |
| estudante_casual | 12,884 | 12.8% | 298 s | -61.0 dBm |
| gamer_social | 12,741 | 15.2% | 362 s | -61.2 dBm |
| leitor_reservado | 10,878 | 11.6% | 274 s | -61.3 dBm |
| maratonista_series | 5,862 | 15.5% | 370 s | -60.8 dBm |

