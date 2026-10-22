# Validação da simulação — OpiniOn (CC2)

Gerado por `python -m opinion.validacao_simulacao` em 22/10/2026 13:27 (28 s). Seção 4.3 do documento do projeto.

> O desempenho de qualquer modelo treinado com estes dados mede a **simulação**, não o comportamento real dos usuários do OpiniOn. Este documento mostra o que foi assumido, se os dados são plausíveis e se as conclusões dependem demais das premissas.

## Resumo

- **Checagens de sanidade:** 16 OK, 1 atenção, 0 falha(s).
- **Sensibilidade:** 7 de 8 conclusões se mantêm nos 3 cenários.
- **Principal premissa a revisar:** a proximidade (RSSI) não influencia a conexão entre desconhecidos na simulação. Sugestão: incluir o RSSI no processo do label na próxima versão do simulador, ou confirmar com os dados reais da semana 16 se ele importa.

## 1. Premissas

Todos os valores estão em `config/simulacao.yaml`. Nenhum vem da pesquisa de monetização (n = 29), que trata de outro comportamento.

| Parâmetro | Valor atual | Justificativa | Como validar |
|---|---|---|---|
| `simulacao.n_usuarios` | 1200 | Tamanho de uma comunidade ativa num campus/cidade média; gera ~55 mil encontros, o bastante para split temporal + validação cruzada com positivos suficientes em cada parte. | Escala do piloto real. |
| `simulacao.semanas` | 12 | Exigido pelo split temporal da seção 4.5 (treino 1–8, validação 9–10, teste 11–12). | — |
| `simulacao.semanas_historico_previo` | 26 | Usuários já têm histórico de avaliações quando a pulseira chega; sem isso a compatibilidade seria vazia nas primeiras semanas. | Histórico médio dos usuários do app. |
| `personas.*.peso` | gamer_social: 0.22; cinefilo: 0.22; leitor_reservado: 0.18; estudante_casual: 0.25; maratonista_series: 0.13 | Mistura provisória de perfis de público cultural/universitário. | Trocar pelas personas do Design. |
| `personas.*.visitas_semana` | gamer_social: {'cinema': 0.6, 'livraria': 0.1, 'campus': 2.5}; cinefilo: {'cinema': 1.6, 'livraria': 0.3, 'campus': 2.0}; leitor_reservado: {'cinema': 0.4, 'livraria': 1.3, 'campus': 2.5}; estudante_casual: {'cinema': 0.5, 'livraria': 0.2, 'campus': 4.0}; maratonista_series: {'cinema': 0.5, 'livraria': 0.2, 'campus': 1.5} | Frequência plausível: estudante vai ao campus quase todo dia; cinéfilo ~1–2 sessões por semana; leitor ~1 livraria por semana. | Pergunta da coleta complementar (seção 3.1). |
| `personas.*.abertura_social` | gamer_social: [5, 3]; cinefilo: [4, 4]; leitor_reservado: [2, 5]; estudante_casual: [3, 4]; maratonista_series: [3, 3] | Variável OCULTA que o modelo não vê (seção 4.3), para o label não ser uma regra redescobrível. Gamers sociais mais abertos, leitores reservados menos. | Interesse em conhecer pessoas (coleta complementar) e taxa de toque nos testes. |
| `locais.*.p_encontro` | cinema: 0.1; livraria: 0.25; campus: 0.03; convencao_games: 0.015 | Chance de dois desconhecidos no mesmo lugar e turno ficarem a menos de ~2 m por tempo suficiente: alta numa livraria pequena, baixa num campus ou pavilhão grandes. | Encontros por hora nos testes reais. |
| `locais.*.duracao_mediana_s` | cinema: 300; livraria: 180; campus: 120; convencao_games: 360 | Fila/saguão de cinema ~5 min, corredor de livraria ~3 min, campus ~2 min, estande de convenção ~6 min. | Distribuição de duração dos encontros reais. |
| `locais.convencao_games.semanas_com_evento` | [3, 7, 11] | Eventos são esporádicos (1 a cada ~4 semanas); cria picos que testam a estabilidade do modelo no tempo. | — |
| `amigos.fracao_em_grupos / p_ir_junto` | 0.5 / 0.3 | Metade das pessoas sai com amigos às vezes; gera encontros longos entre quem já se segue (ruído realista no label). | Proporção de encontros entre quem já se segue nos testes. |
| `modos_descoberta.inicial` | aberto: 0.55; discreto: 0.35; pausado: 0.1 | Maioria aberta ou discreta; ~10% pausada (preocupação com privacidade: 41% na pesquisa anterior citam uso de dados). | Modos escolhidos nos testes. |
| `conexao.intercepto` | -3.9 | Calibra a taxa base: ~5–8% dos encontros entre desconhecidos viram conexão — raro, como esperado para abordagens entre estranhos. | Taxa de conversão real (indicador da seção 3.2). |
| `conexao.compat` | 1.0 | Gosto parecido aumenta a chance de conexão (hipótese do projeto), mas não determina sozinho. | Comparar taxa por faixa de compatibilidade nos dados reais. |
| `conexao.abertura` | 2.2 | Pessoas abertas conectam mais; o produto dos dois exige que ambos estejam abertos. | — |
| `conexao.duracao` | 0.35 | Encontros longos indicam conversa ou atividade em comum. | Taxa por duração nos dados reais. |
| `conexao.contexto` | convencao_games: 0.6; cinema: 0.3; livraria: 0.1; campus: -0.3 | Convenção é social por natureza; campus é casual (seção 3.3). | Taxa por contexto nos testes. |
| `conexao.repeticao` | 0.25 | Encontrar a mesma pessoa várias vezes aumenta a familiaridade. | — |
| `conexao.discreto` | -0.3 | Modo discreto reduz levemente a chance (menos exposição). | — |
| `conexao.ruido` | 0.8 | Fatores não observados (humor, pressa, timidez). Garante que nenhum modelo acerte tudo. | — |
| `conexao.atraso_medio_h` | 14 | Parte das pessoas só abre o app no dia seguinte; ~3–4% das conexões caem depois de 48 h (label 0 mesmo assim). | Tempo até a 1ª interação nos testes. |
| `interacoes_de_fundo.*` | entre_amigos_por_dia: 0.08; entre_seguidos_por_dia: 0.03; aleatorias_por_usuario_dia: 0.15 | Amigos e o feed geram interações independentes de encontros — o label precisa conviver com esse ruído. | Volume de interações por usuário no app real. |
| `toque_interesse.*` | base: 0.03; ganho: 0.5 | Toque mais provável quando há chance de conexão e o usuário é aberto. Dado secundário. | Taxa de toque nos testes do protótipo. |

### Como o label é gerado

Para cada encontro, a chance de virar conexão é

```
p = sigmoide( intercepto + compat·compatibilidade_latente + abertura·(abertura_A × abertura_B)
               + duracao·log(duração/120 s) + efeito_do_local + repeticao·min(encontros anteriores, 5)
               + discreto·[alguém em modo discreto] + ruído )
```

Se sorteado, um dos dois interage com o outro no app depois de um atraso aleatório. O label do pipeline (interação em até 48 h) é calculado depois, a partir das interações — que também incluem interações entre amigos e do feed. A *compatibilidade latente* e a *abertura social* são ocultas: o modelo só vê aproximações (similaridade de notas, atividade, conexões).

## 2. Checagens de sanidade (cenário base)

| Status | Checagem | Valor | Faixa plausível | Por quê |
|---|---|---|---|---|
| ✅ OK | Taxa geral de conexão | 12.3% | 5%–25% | Problema desbalanceado, mas com positivos suficientes para treinar (seção 4.3). |
| ✅ OK | Desconhecidos conectam menos que quem já se segue | 7.0% x 27.3% | desconhecidos < amigos | Amigos interagem no app de qualquer jeito. |
| ✅ OK | Desconhecidos: taxa de conexão | 7.0% | 2%–15% | Abordagem entre estranhos é rara; acima de 15% seria otimista demais. |
| ✅ OK | Duração mediana por local | campus 4.1 min, cinema 7.6 min, convencao_games 7.7 min, livraria 3.5 min | 1–30 min em todos | Encontros de segundos seriam passagens; de horas, só entre amigos. |
| ✅ OK | Nenhum encontro menor que 30 s | mínimo 30 s | >= 30 s | A pulseira descarta contatos curtos (gente só passando). |
| ✅ OK | RSSI médio | -61.4 dBm | -75 a -50 dBm | Equivale a ~0,5–3 m de distância. |
| ✅ OK | Encontros por usuário por semana | 9.2 | 0,5–20 | Plausível para quem frequenta campus e eventos culturais. |
| ✅ OK | Concentração: 10% mais ativos | 22.0% dos encontros | <= 40% | Evita que poucos usuários dominem o dataset (e o modelo). |
| ✅ OK | Campus só em dias úteis | 100.0% | 100% | Premissa de funcionamento. |
| ✅ OK | Convenção só nos fins de semana de evento | sim | semanas [3, 7, 11], sáb/dom | Premissa de calendário. |
| ✅ OK | Encontros entre 8h e 23h59 | 100.0% | 100% | Horário de funcionamento dos locais. |
| ✅ OK | Nenhum encontro com pulseira pausada | 0 encontros | 0 | Modo pausado = pulseira não anuncia (seção 3.3). |
| ✅ OK | Estabilidade semanal da taxa de conexão | desvio 0.6 p.p. | <= 1,5 p.p. | Sem mudança brusca entre treino/validação/teste — o split temporal mede o modelo, não a base. |
| ✅ OK | Encontros repetidos do mesmo par | 28.2% | 10%–60% | Necessário para a feature de encontros anteriores e o k-fold agrupado por par. |
| ✅ OK | Conexões causadas que só aparecem depois de 48 h | 3.2% | 1%–15% | A janela de 48 h perde algumas conexões tardias, como na vida real. |
| ✅ OK | Toque mais frequente quando há conexão | 13.9% x 9.8% | conectou > não conectou | O toque deve carregar algum sinal de interesse. |
| ⚠️ ATENÇÃO | Proximidade (RSSI) influencia conexão entre desconhecidos | AUC 0.510 | AUC >= 0,52 | Esperado no mundo real (quem conversa fica mais perto). Na simulação o RSSI não entra no processo do label — premissa a revisar. |

## 3. Análise de sensibilidade

O dataset foi gerado de novo em três cenários (`config/cenarios_sensibilidade.yaml`). Em cada um rodamos **o mesmo pipeline** e **os mesmos modelos simples**, treinando nas semanas 1–8 e medindo nas semanas 11–12. Os modelos aqui não são otimizados — servem só para ver se as conclusões mudam.

| Cenário | O que muda | Encontros | Taxa de conexão | Entre desconhecidos |
|---|---|---|---|---|
| **pessimista** | Menos conexões, mais ruído, gosto e contexto pesam menos | 64,150 | 12.6% | 3.7% |
| **base** | Premissas atuais (config/simulacao.yaml) | 65,443 | 12.3% | 7.0% |
| **otimista** | Mais conexões, menos ruído, gosto e contexto pesam mais | 65,817 | 15.6% | 14.6% |

### PR-AUC no teste (semanas 11–12) — todos os encontros

| Modelo | pessimista | base | otimista |
|---|---|---|---|
| Regra simples (compatibilidade) | 0.126 | 0.175 | 0.317 |
| Regressão logística | 0.339 | 0.251 | 0.323 |
| Árvore de decisão | 0.341 | 0.252 | 0.373 |
| Random Forest | 0.356 | 0.259 | 0.422 |
| *prevalência (acaso)* | *0.124* | *0.128* | *0.172* |

### PR-AUC no teste — só desconhecidos (o público real da pulseira)

| Modelo | pessimista | base | otimista |
|---|---|---|---|
| Regra simples (compatibilidade) | 0.058 | 0.155 | 0.380 |
| Regressão logística | 0.064 | 0.185 | 0.449 |
| Árvore de decisão | 0.062 | 0.174 | 0.429 |
| Random Forest | 0.063 | 0.200 | 0.477 |
| *prevalência (acaso)* | *0.039* | *0.077* | *0.162* |

### Taxa de conexão entre desconhecidos por contexto

| Contexto | pessimista | base | otimista |
|---|---|---|---|
| convencao_games | 5.6% | 11.0% | 23.6% |
| cinema | 4.8% | 9.8% | 20.7% |
| livraria | 3.4% | 6.0% | 13.2% |
| campus | 2.3% | 4.0% | 7.1% |

### As conclusões se mantêm?

| Conclusão | Mantém nos 3 cenários? | Detalhe |
|---|---|---|
| Todo modelo supera a regra simples (todos os encontros) | ✅ sim | pessimista: regra 0.126; base: regra 0.175; otimista: regra 0.317 |
| Todo modelo supera a regra simples (só desconhecidos) | ✅ sim |  |
| O melhor modelo supera a regra simples (só desconhecidos) | ✅ sim | pessimista: 0.064 x regra 0.058; base: 0.200 x regra 0.155; otimista: 0.477 x regra 0.380 |
| O melhor modelo é o mesmo | ✅ sim | pessimista: Random Forest (+0.015 sobre o 2º); base: Random Forest (+0.007 sobre o 2º); otimista: Random Forest (+0.049 sobre o 2º) |
| Ordem dos contextos entre desconhecidos é a mesma | ✅ sim | pessimista: convencao_games > cinema > livraria > campus; base: convencao_games > cinema > livraria > campus; otimista: convencao_games > cinema > livraria > campus |
| Convenção converte mais que campus (desconhecidos) | ✅ sim | pessimista: 2.4×; base: 2.7×; otimista: 3.3× |
| As 3 features mais importantes (Random Forest) se repetem (≥ 2 de 3) | ❌ não | pessimista: seguem_mutuamente, algum_segue, horas_desde_ultimo_encontro_par; base: seguem_mutuamente, sim_generos, algum_segue; otimista: sim_generos, compat_score, log_duracao |
| Problema continua desbalanceado (3%–30% positivos) | ✅ sim | 12.6%, 12.3%, 15.6% |

**Leitura.** Os números absolutos mudam bastante entre cenários (é esperado: as premissas mudam a dificuldade do problema). O que importa é que a **comparação** se sustente — se uma conclusão só vale no cenário base, ela não deve ser apresentada como resultado do projeto.

Repare que o cenário pessimista tem PR-AUC *geral* mais alta que o base: nele os amigos interagem mais, e o modelo acerta reconhecendo amigos. Por isso a métrica só entre desconhecidos é a que conta para a pulseira.

**O que não se manteve e o que isso significa:**

- *As 3 features mais importantes (Random Forest) se repetem (≥ 2 de 3).* A importância das features depende das premissas: quando a compatibilidade pesa mais (otimista), as features de gosto sobem ao topo; quando pesa menos (pessimista), dominam as de relação prévia. Portanto, rankings de importância obtidos nestes dados NÃO devem ser apresentados como descoberta sobre os usuários reais — apenas como consequência das premissas.

## 4. Comparação com dados reais (semana 16)

Quando o protótipo do Design for testado com usuários, os encontros reais entram pelo mesmo MQTT e pelo mesmo pipeline. Comparar com a simulação:

| O que comparar | Como | Se divergir |
|---|---|---|
| Duração dos encontros por contexto | mediana e teste de Kolmogorov–Smirnov | ajustar `duracao_mediana_s` |
| RSSI médio | média e histograma | ajustar o modelo de rádio da pulseira virtual |
| Encontros por pessoa por hora | contagem | ajustar `p_encontro` |
| Taxa de conexão (geral e por contexto) | proporção com intervalo de confiança | ajustar `intercepto` e `contexto` |
| Taxa de toque e de toque mútuo | proporção | ajustar `toque_interesse` |
| Fração de encontros entre quem já se segue | proporção | ajustar `amigos` |

Com poucos participantes nos testes, a comparação será qualitativa (mesma ordem de grandeza? mesma direção dos efeitos?). Ainda assim, ela é a única evidência externa de que a simulação é razoável.

## 5. Limitações conhecidas

- Personas, frequências e efeitos são premissas — as personas reais do Design ainda não foram incorporadas.
- O RSSI não influencia a conexão entre desconhecidos (ver checagem com ⚠️).
- Dia da semana e horário não têm efeito próprio no label (só via local).
- A rede de amigos é fixa: não surgem novos grupos ao longo das 12 semanas.
- A taxa de silenciamento não reage aos sinais (a pulseira ainda não sinaliza).
