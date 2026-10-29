# Relatório da simulação — bootstrap dataset OpiniOn (CC1)

Semente 42 · 1200 usuários · 12 semanas (06/07/2026 a 27/09/2026) · todos os parâmetros são premissas da equipe (`config/simulacao.yaml`).

> Dados **sintéticos**. Qualquer desempenho medido com eles é desempenho sobre a simulação, não sobre o comportamento real dos usuários.

## Volume

| Item | Quantidade |
|---|---|
| usuarios | 1,200 |
| obras | 48,018 |
| avaliacoes | 58,251 |
| seguidores | 6,648 |
| interacoes | 43,280 |
| encontros | 66,344 |
| toques_interesse | 7,172 |
| trocas_de_modo | 2,344 |

## Label `conectou` (interação em até 48 h)

- Encontros com label positivo: **12.3%** (problema desbalanceado, como esperado)
- Encontros que de fato *causaram* conexão (variável oculta): 5.3%
- Conexões causadas cuja 1ª interação veio depois de 48 h (label = 0 mesmo assim): 3.2%
- Positivos entre amigos (que já se seguem): 27.6% · entre desconhecidos: 7.0%

A diferença entre *conectou* e *causou* é intencional: amigos interagem no app de qualquer jeito, e algumas conexões só acontecem depois da janela. O modelo precisa lidar com esse ruído.

## Por contexto

| Local | Encontros | % conectou | Duração mediana | RSSI médio |
|---|---|---|---|---|
| campus | 18,804 | 14.2% | 244 s | -60.4 dBm |
| cinema | 13,770 | 15.2% | 455 s | -61.1 dBm |
| convencao_games | 7,908 | 13.9% | 460 s | -61.6 dBm |
| livraria | 25,862 | 8.8% | 210 s | -62.2 dBm |

## Por semana (para o split temporal 1–8 / 9–10 / 11–12)

| Semana | Encontros | % conectou |
|---|---|---|
| 1 | 5,277 | 12.3% |
| 2 | 5,085 | 12.4% |
| 3 | 8,067 | 12.7% |
| 4 | 4,911 | 11.9% |
| 5 | 4,842 | 12.3% |
| 6 | 4,947 | 11.8% |
| 7 | 7,174 | 13.1% |
| 8 | 4,852 | 11.4% |
| 9 | 4,647 | 11.1% |
| 10 | 4,854 | 12.1% |
| 11 | 7,129 | 13.2% |
| 12 | 4,559 | 12.2% |

Picos nas semanas 3, 7 e 11 = fins de semana com convenção de games.

## Por persona de quem publicou

| Persona | Encontros | % conectou | Duração mediana | RSSI médio |
|---|---|---|---|---|
| cinefilo | 19,125 | 14.3% | 324 s | -61.3 dBm |
| estudante_casual | 8,121 | 10.7% | 280 s | -61.1 dBm |
| gamer_social | 8,715 | 13.9% | 328 s | -61.2 dBm |
| leitor_reservado | 29,999 | 11.0% | 245 s | -61.6 dBm |
| maratonista_series | 384 | 11.7% | 394 s | -60.8 dBm |

