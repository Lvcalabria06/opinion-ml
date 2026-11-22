# Relatório da simulação — bootstrap dataset OpiniOn (CC1)

Semente 42 · 1892 usuários · 12 semanas (06/07/2026 a 27/09/2026) · todos os parâmetros são premissas da equipe (`config/simulacao.yaml`).

> Os **encontros** são simulados (local, hora, duração e RSSI são premissas da equipe); usuários, gostos e o **rótulo** (amizade no Last.fm) são reais. Os encontros não olham a amizade; o desempenho medido é sobre essa combinação, não sobre o comportamento real em campo.

## Volume

| Item | Quantidade |
|---|---|
| usuarios | 1,892 |
| obras | 17,632 |
| avaliacoes | 92,834 |
| seguidores | 25,434 |
| interacoes | 0 |
| encontros | 162,897 |
| toques_interesse | 9,791 |
| trocas_de_modo | 3,689 |

## Label `conectou` (amizade real no Last.fm)

- Encontros com label positivo (os dois já são amigos no Last.fm): **0.8%** (1,235 encontros, 1,171 pares) — problema desbalanceado, como a rede de amizades da população
- O rótulo é estático: não depende da data do encontro nem de nenhuma janela; a divisão treino/validação/teste é feita por par de usuários (D-014).

## Por contexto

| Local | Encontros | % conectou | Duração mediana | RSSI médio |
|---|---|---|---|---|
| campus | 63,799 | 0.7% | 120 s | -62.9 dBm |
| cinema | 31,469 | 0.8% | 299 s | -62.9 dBm |
| convencao_games | 33,396 | 0.9% | 362 s | -62.9 dBm |
| livraria | 34,233 | 0.8% | 180 s | -62.9 dBm |

## Por semana

| Semana | Encontros | % conectou |
|---|---|---|
| 1 | 11,493 | 0.7% |
| 2 | 11,220 | 0.7% |
| 3 | 22,910 | 0.8% |
| 4 | 11,203 | 0.9% |
| 5 | 10,624 | 0.7% |
| 6 | 11,092 | 0.7% |
| 7 | 21,669 | 0.8% |
| 8 | 10,648 | 0.7% |
| 9 | 10,729 | 0.6% |
| 10 | 10,339 | 0.8% |
| 11 | 20,829 | 0.8% |
| 12 | 10,141 | 0.7% |

Picos nas semanas 3, 7 e 11 = fins de semana com convenção de games.

## Por persona de quem publicou

| Persona | Encontros | % conectou | Duração mediana | RSSI médio |
|---|---|---|---|---|
| cinefilo | 35,706 | 0.8% | 219 s | -62.9 dBm |
| estudante_casual | 39,291 | 0.7% | 162 s | -62.8 dBm |
| gamer_social | 39,646 | 0.8% | 239 s | -62.9 dBm |
| leitor_reservado | 34,496 | 0.7% | 169 s | -62.9 dBm |
| maratonista_series | 13,758 | 0.8% | 208 s | -62.8 dBm |

