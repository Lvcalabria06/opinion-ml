# Relatório da simulação — bootstrap dataset OpiniOn (CC1)

Semente 42 · 1200 usuários · 12 semanas (06/07/2026 a 27/09/2026) · todos os parâmetros são premissas da equipe (`config/simulacao.yaml`).

> Dados **sintéticos**. Qualquer desempenho medido com eles é desempenho sobre a simulação, não sobre o comportamento real dos usuários.

## Volume

| Item | Quantidade |
|---|---|
| usuarios | 1,200 |
| obras | 48,018 |
| avaliacoes | 58,251 |
| seguidores | 5,670 |
| interacoes | 38,426 |
| encontros | 69,384 |
| toques_interesse | 4,443 |
| trocas_de_modo | 2,343 |

## Label `conectou` (interação em até 48 h)

- Encontros com label positivo: **8.1%** (problema desbalanceado, como esperado)
- Encontros que de fato *causaram* conexão (variável oculta): 0.8%
- Conexões causadas cuja 1ª interação veio depois de 48 h (label = 0 mesmo assim): 3.1%
- Positivos entre amigos (que já se seguem): 27.6% · entre desconhecidos: 1.2%

A diferença entre *conectou* e *causou* é intencional: amigos interagem no app de qualquer jeito, e algumas conexões só acontecem depois da janela. O modelo precisa lidar com esse ruído.

## Por contexto

| Local | Encontros | % conectou | Duração mediana | RSSI médio |
|---|---|---|---|---|
| campus | 19,267 | 12.9% | 262 s | -60.2 dBm |
| cinema | 13,495 | 9.3% | 463 s | -61.1 dBm |
| convencao_games | 8,029 | 6.9% | 459 s | -61.6 dBm |
| livraria | 28,593 | 4.6% | 209 s | -62.1 dBm |

## Por semana (para o split temporal 1–8 / 9–10 / 11–12)

| Semana | Encontros | % conectou |
|---|---|---|
| 1 | 5,238 | 7.6% |
| 2 | 5,470 | 8.2% |
| 3 | 8,305 | 8.3% |
| 4 | 5,103 | 8.1% |
| 5 | 5,147 | 7.7% |
| 6 | 4,869 | 8.9% |
| 7 | 8,043 | 7.7% |
| 8 | 5,138 | 8.2% |
| 9 | 4,915 | 8.2% |
| 10 | 5,186 | 8.8% |
| 11 | 7,142 | 7.5% |
| 12 | 4,828 | 8.7% |

Picos nas semanas 3, 7 e 11 = fins de semana com convenção de games.

## Por persona de quem publicou

| Persona | Encontros | % conectou | Duração mediana | RSSI médio |
|---|---|---|---|---|
| cinefilo | 19,261 | 8.4% | 318 s | -61.3 dBm |
| estudante_casual | 8,507 | 9.0% | 290 s | -61.1 dBm |
| gamer_social | 8,508 | 9.1% | 344 s | -61.2 dBm |
| leitor_reservado | 32,660 | 7.4% | 248 s | -61.5 dBm |
| maratonista_series | 448 | 12.9% | 418 s | -60.1 dBm |

