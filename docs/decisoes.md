# Registro de decisões

Cada decisão que muda o que se constrói entra aqui, com data, motivo e consequência. A mais recente
fica em cima. Decisões ainda abertas ficam em [`status.md`](status.md).

---

## D-010 · 29/10/2026 · Baseline CC3: escolha e aprovação na validação, limiar por F0,5

- **Decisão:** a escolha entre Regressão Logística e Árvore de Decisão usa PR-AUC entre
  desconhecidos na validação, e é **na validação** que se exige superar a regra simples
  (`compat_score`). O limiar do sinal maximiza **F0,5** (precisão pesa o dobro do recall:
  sinal errado cansa o usuário) entre desconhecidos da validação. A regra simples recebe
  o mesmo protocolo: o limiar dela também é escolhido na validação.
- **Por quê:** aprovar no teste e rodar de novo até passar transforma o teste em validação.
  O teste agora só é relatado; nenhuma escolha usa ele.
- **Métricas registradas:** PR-AUC e ROC-AUC (todos e desconhecidos), Brier, e no limiar
  precisão, recall, F0,5, taxa de sinal e matriz de confusão; curvas PR, ROC, calibração e
  matriz de confusão como artefatos no MLflow.
- **Checkpoint 2:** carrega o modelo do registry (não retreina) e pontua um encontro
  **novo** no formato do contrato MQTT, passando pelas mesmas funções do pipeline.

## D-009 · 22/10/2026 · Um dataset unificado: Amazon Reviews 2023

- **Contexto:** o professor aceitou dados reais de outras empresas no lugar do catálogo inventado.
  Juntar ou separar por mídia ficou a critério da equipe.
- **Decisão:** um único dataset, [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
  (McAuley Lab, UCSD). Mesmos usuários reais avaliando livro, filme/série, jogo e música.
  1.200 pessoas sorteadas com cota por mídia dominante; janela real 2021-01-04 a 2021-09-27,
  deslocada 261 semanas para 2026. Detalhes em [`fontes_dados.md`](fontes_dados.md).
- **Por que não Spotify+Netflix+Steam:** isso inventaria que pessoas diferentes são a mesma.
- **O que continua simulado:** encontros da pulseira, rede de seguidores, interações no app, label.

## D-008 · 21/10/2026 · Trocar dados inventados por datasets públicos reais

- **Contexto:** o professor avisou que usar dataset inventado perde pontos.
- **Decisão:** usar dados reais de usuários publicados por outras empresas (música, filmes, jogos,
  livros etc.), filtrados para o caso do OpiniOn, com a fonte citada. Juntar tudo ou separar por mídia
  fica a critério da equipe.
- **Consequência:** catálogo, avaliações e usuários passam a vir de fontes reais. Encontros e label
  continuam dependendo de simulação enquanto não houver pulseira. Detalhes e decisões abertas na
  Frente A de [`status.md`](status.md).

## D-007 · 14/10/2026 · Não apresentar "melhor algoritmo" nem ranking de features como descoberta

- **Contexto:** na análise de sensibilidade, o melhor modelo e as features mais importantes mudam
  conforme as premissas da simulação.
- **Decisão:** só apresentar como resultado o que se mantém nos 3 cenários (ordem dos contextos;
  modelo supera a regra simples). A escolha de modelo (CC4) se apoia em validação cruzada,
  explicabilidade e calibração.

## D-006 · Sprint 2 · Métrica principal: PR-AUC, também só entre desconhecidos

- **Motivo:** o problema é desbalanceado (~14% positivos), e amigos inflam a métrica geral. O público
  da pulseira são os desconhecidos.
- **Consequência:** todo resultado reporta PR-AUC geral, PR-AUC com `algum_segue == 0`, a prevalência
  (acaso) e a regra simples. A métrica final depende do custo do sinal errado (aberto com o Design).

## D-005 · Sprint 2 · Padronização fica no pipeline do modelo, não na preparação

- **Motivo:** ajustar média e desvio em todo o dataset vaza informação do teste.
- **Consequência:** `dataset_modelo.parquet` sai sem padronização; o `StandardScaler` vai dentro do
  pipeline do sklearn, ajustado só no treino (ou no treino de cada fold).

## D-004 · Sprint 2 · Split temporal com purga da fronteira

- **Decisão:** treino semanas 1–8, validação 9–10, teste 11–12 (seção 4.5). Encontros do treino cuja
  janela de 48 h invade a validação são removidos (532 encontros).
- **Consequência:** nunca usar split aleatório. O k-fold do CC6 é agrupado por `par_id`.

## D-003 · Sprint 2 · Nenhuma feature olha para o futuro

- **Decisão:** toda feature de um encontro usa só dados de antes do seu início.
- **Garantia:** `tests/test_pipeline.py::test_sem_vazamento_do_futuro` recalcula as features apagando
  o futuro e exige resultado idêntico. Feature nova precisa passar nesse teste.

## D-002 · Sprint 1 · IDs anônimos e rotativos na pulseira

- **Decisão:** a pulseira anuncia `u-` + HMAC-SHA256(segredo, usuário|data)[:10], que muda todo dia.
  Só o backend, com `OPINION_SEGREDO_TOKEN`, traduz.
- **Consequência:** o dataset em `data/bootstrap/` foi gerado com o segredo de desenvolvimento do
  `.env.example`; trocar o segredo exige regerar os dados.

## D-001 · Sprint 1 · Label com variáveis ocultas e ruído

- **Decisão:** o simulador decide as conexões por um processo probabilístico com variáveis ocultas
  (abertura social, compatibilidade "verdadeira") e ruído, para o label não ser uma regra que o modelo
  redescobre.
- **Consequência:** essas variáveis ficam em `verdade_simulacao_NAO_USAR_COMO_FEATURE/` e nunca entram
  no modelo.
