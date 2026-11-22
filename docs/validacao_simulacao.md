# Validação da simulação — OpiniOn (CC2, experimento Last.fm)

Gerado por `python -m opinion.validacao_simulacao` em 22/11/2026 17:27 (0 s).

> Neste experimento o **rótulo é real**: `conectou = 1` quando os dois usuários são amigos no Last.fm (HetRec 2011). Usuários, artistas, escutas e amizades são dados reais; **só os encontros e a pulseira são simulados**, e de forma independente da amizade. O desempenho do modelo mede, portanto, o quanto gosto musical e contexto do encontro explicam uma amizade real, sob a premissa de que os encontros são sorteados sem olhar a rede. Não é evidência de comportamento de encontros reais.

## Resumo

- **Checagens de sanidade dos encontros:** 13 OK, 0 atenção, 0 falha(s).
- **Não se aplica:** o processo de conexão simulado, as interações de fundo, o toque como sinal do label e a análise de sensibilidade por cenários (`config/cenarios_sensibilidade.yaml`) só valiam para o rótulo simulado (`interacao_48h`). Com `label.tipo: amizade` não há premissa de conexão a testar; a premissa que importa é a independência entre encontro e amizade, verificada abaixo.

## 1. Premissas dos encontros

Valores em `config/simulacao.yaml`. Gosto, usuários e amizades vêm do Last.fm (`docs/fontes_dados.md`).

| Parâmetro | Valor atual | Justificativa | Como validar |
|---|---|---|---|
| `personas.*.peso` | gamer_social: 0.22; cinefilo: 0.22; leitor_reservado: 0.18; estudante_casual: 0.25; maratonista_series: 0.13 | Mistura provisória de perfis de público cultural/universitário; só decide onde cada usuário circula (o gosto musical vem do Last.fm). | Trocar pelas personas do Design. |
| `personas.*.visitas_semana` | gamer_social: {'cinema': 0.6, 'livraria': 0.1, 'campus': 2.5}; cinefilo: {'cinema': 1.6, 'livraria': 0.3, 'campus': 2.0}; leitor_reservado: {'cinema': 0.4, 'livraria': 1.3, 'campus': 2.5}; estudante_casual: {'cinema': 0.5, 'livraria': 0.2, 'campus': 4.0}; maratonista_series: {'cinema': 0.5, 'livraria': 0.2, 'campus': 1.5} | Frequência plausível por tipo de local, mantida nos valores originais da simulação. | Pergunta da coleta complementar (seção 3.1). |
| `locais.*.p_encontro` | cinema: 0.3; livraria: 0.75; campus: 0.09; convencao_games: 0.045 | Chance de dois desconhecidos no mesmo lugar e turno ficarem a menos de ~2 m por tempo suficiente. No experimento foi multiplicada por 3 (valores originais 0,10/0,25/0,03/0,015) para haver amigos encontrados suficientes (positivos). | Encontros por hora nos testes reais. |
| `locais.*.duracao_mediana_s` | cinema: 300; livraria: 180; campus: 120; convencao_games: 360 | Fila de cinema ~5 min, livraria ~3 min, campus ~2 min, estande de convenção ~6 min. | Distribuição de duração dos encontros reais. |
| `locais.convencao_games.semanas_com_evento` | [3, 7, 11] | Eventos esporádicos; criam picos que testam a estabilidade no tempo. | — |
| `modos_descoberta.inicial` | aberto: 0.55; discreto: 0.35; pausado: 0.1 | Maioria aberta ou discreta; ~10% pausada (privacidade). | Modos escolhidos nos testes. |

## 2. Checagens de sanidade

| Status | Checagem | Valor | Faixa plausível | Por quê |
|---|---|---|---|---|
| ✅ OK | Amigos entre os pares que se encontraram ~ densidade da população (independência) | 0.76% x 0.71% (razão 1.08) | razão entre 0,85 e 1,15 | Os encontros são simulados sem olhar a amizade; se a fração fosse muito diferente da densidade, o simulador estaria vazando amizade para o encontro (ou o contrário). Faixa de ±15%: com ~1,2 mil pares de amigos entre os pares que se encontraram, o desvio amostral esperado da razão é ~3%, então 15% já são ~5 desvios. |
| ✅ OK | Taxa geral de positivos (encontros) | 0.76% | 0,2%–40% (faixa da DAG) | Desbalanceado como a rede de amizades da população; precisa de positivos suficientes por parte do split. |
| ✅ OK | Duração mediana por local | campus 2.0 min, cinema 5.0 min, convencao_games 6.0 min, livraria 3.0 min | 1–30 min em todos | Encontros de segundos seriam passagens; de horas, só entre amigos. |
| ✅ OK | Nenhum encontro menor que 30 s | mínimo 30 s | >= 30 s | A pulseira descarta contatos curtos (gente só passando). |
| ✅ OK | RSSI médio | -62.9 dBm | -75 a -50 dBm | Equivale a ~0,5–3 m de distância. |
| ✅ OK | Encontros por usuário por semana | 14.3 | 0,5–20 | Plausível para quem frequenta campus e eventos culturais. |
| ✅ OK | Concentração: 10% mais ativos | 15.2% dos encontros | <= 40% | Evita que poucos usuários dominem o dataset (e o modelo). |
| ✅ OK | Campus só em dias úteis | 100.0% | 100% | Premissa de funcionamento. |
| ✅ OK | Convenção só nos fins de semana de evento | sim | semanas [3, 7, 11], sáb/dom | Premissa de calendário. |
| ✅ OK | Encontros entre 8h e 23h59 | 100.0% | 100% | Horário de funcionamento dos locais. |
| ✅ OK | Nenhum encontro com pulseira pausada | 0 encontros | 0 | Modo pausado = pulseira não anuncia (seção 3.3). |
| ✅ OK | Estabilidade semanal da taxa de conexão | desvio 0.1 p.p. | <= 1,5 p.p. | Sem mudança brusca ao longo das semanas — a taxa estável indica que a divisão por pares mede o modelo, não uma deriva da base. |
| ✅ OK | Encontros repetidos do mesmo par | 5.9% | 2%–60% | Com o split por par a repetição não vaza entre treino e teste; só alimenta a feature de encontros anteriores. Faixa mais baixa que a do rótulo simulado (10%) porque com ~1,9 mil usuários os pares são muitos e raros de repetir. |

## 3. Limitações conhecidas

- Os encontros são sorteados sem olhar a amizade: na vida real amigos se encontram mais, então o sinal de proximidade/duração/repetição para amizade é subestimado aqui.
- O rótulo é estático (amizade já existente no dataset), não a formação de uma amizade após o encontro.
- O RSSI e o horário não têm relação com o rótulo por construção.
- Personas, frequências e locais são premissas; as personas reais do Design não foram incorporadas.
- Sem análise de sensibilidade: com rótulo real não há parâmetro de conexão a variar.
