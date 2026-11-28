# Registro de decisões

Cada decisão que muda o que se constrói entra aqui, com data, motivo e consequência. A mais recente
fica em cima. Decisões ainda abertas ficam em [`status.md`](status.md).

---

## D-018 · 28/11/2026 · Docker Compose com servidor MLflow, treino na DAG e espera da ingestão (CC8)

- **Decisão:** o compose ganha um servidor MLflow com proxy de artefatos; Airflow, API e consumidor
  falam com ele por HTTP (`MLFLOW_TRACKING_URI`). A DAG ganha a task `treino`. O serviço do replay só
  termina depois que a ingestão grava todos os encontros (`opinion.aguardar_ingestao`).
- **Motivo:** com SQLite + volume, o caminho dos artefatos gravado por um container não existe nos
  outros. No teste de 28/11 a ingestão terminou ~3 min depois da pulseira (163 mil encontros): sem a
  espera, o Airflow leria o banco pela metade.
- **Consequência:** `scikit-learn` fixado na mesma versão nas duas imagens (a API carrega o pickle
  treinado pelo Airflow).

## D-017 · 28/11/2026 · Inferência após o encontro: chance de amizade, sinal pelo tópico feedback (CC7)

- **Decisão:** o aviso de afinidade continua na proximidade (D-013). No **fim** do encontro, o consumidor
  de inferência estima a chance de amizade com o modelo de produção e, acima do limiar, publica
  `feedback` com `tipo: previsao_amizade`. Faixas: `periferico_leve` a partir do limiar,
  `periferico_perceptivel` a partir de 1,5 × limiar. Pulseira pausada não recebe sinal.
- **Modelo de produção:** a regressão logística do CC5 (`C≈2e-5`), recomendada no CC6 (D-015),
  calibrada com sigmoide na validação para a probabilidade poder ser mostrada como chance. A DAG só
  promove uma versão nova se ela superar a regra simples na validação.
- **Consequência:** a "taxa de conversão" e a "precisão dos sinais" do dashboard usam o mesmo rótulo
  (par amigo na rede). Como ele é estático, os indicadores medem acerto do sinal, não amizades criadas
  pelo encontro.

## D-016 · 28/11/2026 · CC4 com sete famílias (KNN, AdaBoost e MLP)

- **Decisão:** a comparação do Last.fm inclui KNN, AdaBoost e MLP, para cumprir os três modelos da lista
  da especificação (antes havia só Árvore e Random Forest da lista).
- **Resultado:** Random Forest segue escolhida na validação (0,061); KNN 0,051, AdaBoost 0,051 e MLP 0,043
  na validação. No teste os sete ficam entre 0,039 e 0,063, com intervalos sobrepostos; todos superam a
  regra simples (0,027). O CC5 e o CC6 (D-015) não mudam.

## D-015 · 23/11/2026 · CC5 (Optuna) e CC6 (validação cruzada) no experimento Last.fm

- **CC5:** Optuna (TPE, semente 42) otimiza regressão logística (50 tentativas) e Random Forest (30),
  objetivo PR-AUC na validação; limiar F0,5 na validação; teste medido uma vez. Espaços em
  `config/modelo.yaml → otimizacao`. Resultado: o melhor ponto fica na borda de regularização máxima
  (LR `C≈2e-5`; RF profundidade 3) e o ganho sobre a grade do CC4 na validação não se transferiu ao
  teste (LR 0,061; RF 0,053; regra 0,027; prevalência 0,0075). O espaço não foi ampliado mais por isso.
- **CC6:** `StratifiedGroupKFold` (k = 5, grupos = `par_id`, 3 sementes) só sobre treino + validação.
  LR 0,061 ± 0,008 e RF 0,058 ± 0,007 de PR-AUC são indistinguíveis; RF tem gap treino−validação maior
  (+0,018 contra −0,002); regra 0,027 ± 0,003. Leave-One-Out não aplicado (~138 mil treinos e vazamento
  entre encontros do mesmo par). Curvas de aprendizado e complexidade com 1 semente.
- **Consequência:** com desempenho igual, a regressão logística é o modelo recomendado (mais simples,
  explicável e sem sobreajuste); o vencedor formal do CC5 (RF) fica registrado em `opinion-cc5-lastfm`.

## D-014 · 22/11/2026 · Experimento Last.fm: prever amizade (branch `experimento-lastfm`)

- **Decisão:** trocar, só nesta branch, a fonte por HetRec 2011 Last.fm e a pergunta por "dois usuários
  se encontraram; qual a chance de serem amigos?". O label é a amizade real (estática, sem data).
  Encontros continuam simulados, independentes da amizade (`p_encontro` ×3: 162.897 encontros,
  153.212 pares, 1.171 pares de amigos, 0,76% positivos contra densidade 0,71% da população).
- **Por quê:** a Amazon dá pouca sobreposição (~2% dos pares sorteados dividem um item); no Last.fm,
  56,5% dividem ≥1 artista. Amigos: 92,2%; Jaccard médio de artistas 0,103 contra 0,024 em não amigos.
  Além disso o label passa a ser real, não simulado.
- **Protocolo:** split por par 70/15/15 estratificado (semente 7), pois não há datas de amizade.
  Fora das features: `algum_segue`, `seguem_mutuamente`, `interacoes_anteriores_par`, `interagiram_antes`,
  `conexoes_min`, `conexoes_max`. Métrica: PR-AUC sobre todos os encontros (sem recorte "desconhecidos"),
  ao lado da prevalência, da regra simples e do limiar F0,5.
- **Consequências:** `seguidores.csv` passa a ser a amizade real; sem split temporal e com possível
  causalidade reversa. Os encontros não carregam sinal real do label, então o resultado vem do gosto.
  Resultados (teste, 184 encontros positivos em 176 pares, prevalência 0,0075): regressão logística PR-AUC 0,063, regra
  simples 0,027, Jaccard de artistas sozinho 0,057 (medição pontual da equipe no mesmo split de teste, fora do pipeline versionado); CC4 escolheu Random Forest na validação (teste 0,055).
  Requer validação do professor.

## D-013 · 18/11/2026 · Aviso por mídia no início do encontro

O aviso indica os assuntos em que duas pessoas próximas mostram alta afinidade.
A decisão usa apenas os históricos daquela mídia disponíveis no início do encontro.
A pulseira publica a proximidade de imediato e registra a duração ao encerrar o encontro.
Os modelos estimam a interação posterior com os dados disponíveis ao fim. O corte
inicial de 0,68 é uma hipótese de produto a validar com participantes.

O treino atual recorta os encontros por histórico bilateral em cada mídia, incluindo afinidades
baixas. Os cinco registros CC4 usam `opinion-cc4-aviso-categoria-<midia>@melhor`. A decisão
de proximidade é independente da probabilidade estimada para interação após o encontro.

## D-012 · 17/11/2026 · Estrutura do repositório conforme o item 6 da especificação

- **Decisão:** o pacote `opinion` vai para `src/opinion/` (instalável por `pyproject.toml`, e o
  `pip install -r requirements-analise.txt` já o instala); a pulseira virtual vai para `embedded/`
  (`python -m embedded.pulseira_virtual`); a DAG para `dags/`; o MLflow (banco e artefatos) para
  `mlruns/`; `api/`, `app/` e `reports/` criadas com README até o CC7, o CC9 e o relatório.
- **Por que separar a pulseira:** quem corrige encontra o "simulador do sistema embarcado" na pasta que
  a especificação nomeia. O contrato MQTT (`src/opinion/contrato.py`) fica em `src/` porque a ingestão
  do backend também o usa.
- **Mantidas como ajuste permitido:** `config/`, `docs/`, `tests/`, `infra/`.
- **Raiz do projeto:** `OPINION_HOME`, ou duas pastas acima de `src/opinion/`. No Docker, o pacote roda
  sem instalar (`PYTHONPATH=/app/src`, `OPINION_HOME=/app`).

## D-011 · 29/10/2026 · CC4: comparação no mesmo protocolo, com intervalo de confiança

- **Decisão:** quatro famílias (Regressão logística, Árvore de decisão, Random Forest, Gradient
  Boosting), cada uma com grade pequena e fixa em `config/modelo.yaml → comparacao`. Configuração,
  limiar (F0,5) e família vencedora escolhidos na validação; teste medido uma vez por família.
- **Incerteza:** diferenças de PR-AUC entre desconhecidos no teste com IC 95% por bootstrap pareado
  (1.000 reamostras). Diferença cujo IC inclui zero não é apresentada como "modelo melhor".
- **Fora do escopo de propósito:** otimização de hiperparâmetros (CC5, Optuna) e validação cruzada
  e estabilidade (CC6). Calibração é medida (Brier, curva), mas não corrigida aqui.
- **Registro histórico:** nessa rodada geral, o vencedor foi registrado como `opinion-cc4-melhor@melhor`
  e o baseline como `opinion-baseline-cc3@champion`. Os registros atuais estão na D-013.

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
