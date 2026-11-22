# Experimento Last.fm — design (aprovado em 22/11/2026)

## Intenção
Trocar a fonte do gosto (Amazon, 5 mídias) pelo **HetRec 2011 Last.fm** (música) e o rótulo
simulado ("interagiu em 48 h") por um rótulo **real**: os dois usuários são **amigos** no Last.fm.
Pergunta do modelo: *dois usuários se encontraram; qual a chance de eles serem amigos?*
A pulseira continua igual (detecta o encontro, publica via MQTT, recebe o sinal).
Branch `experimento-lastfm`, a partir da `main` (310d1a2). **Sem commits do Claude.** Parar no CC4.

## Por que (medido em 22/11/2026 sobre os arquivos baixados)
- 1.892 usuários, 17.632 artistas, 92.834 pares usuário–artista, 12.717 amizades (densidade 0,71%).
- 56,5% dos pares aleatórios têm ≥1 artista em comum (Amazon: ~2%); amigos: 92,2%.
- Jaccard médio de artistas: amigos 0,103 × não-amigos 0,024. Só o Jaccard, 1 amigo para 20
  não-amigos: ROC-AUC 0,83, PR-AUC 0,25 (acaso 0,05).

## Decisões
| Tema | Decisão |
|---|---|
| Gosto | artista = obra (`midia = musica`); nota 1–5 por quintis das escutas **de cada usuário**; gêneros = tags mais usadas do artista mapeadas pelos `generos` de `config/fontes.yaml`; avaliações datadas antes do início da simulação (perfil prévio) |
| Datas das tags | não usadas (há anos inválidos, ex.: 1956) |
| Encontros | simulados **independentes da amizade**: sem grupos de amigos, sem "amigos vão juntos"; mais encontros que hoje para haver positivos suficientes |
| Rótulo | `conectou = 1` se o par é amigo em `user_friends` (simétrico); sempre disponível |
| Rede no app | `seguidores` = amizades reais (as duas direções), datadas antes do início |
| Features excluídas | `algum_segue`, `seguem_mutuamente`, `interacoes_anteriores_par`, `interagiram_antes`, `conexoes_min/max` (são o rótulo ou derivam dele) |
| Divisão | por **par**: cada par inteiro em treino, validação ou teste (70/15/15, estratificado pelo rótulo, semente fixa) |
| Métrica | PR-AUC geral (o recorte "desconhecidos" deixa de existir), prevalência, regra simples (`compat_score`), limiar por F0,5 |
| MLflow | experimentos e modelos com sufixo `-lastfm` |
| Escopo | CC1 a CC4 funcionando (pulseira/MQTT/ingestão, pipeline, EDA, baseline, Checkpoint 2, comparação); validação da simulação reduzida às checagens dos encontros |

## Fora do escopo
CC5, CC6, Docker, commits. A `main` não é alterada.

## Riscos aceitos
- Sem data das amizades: o rótulo é estático; não dá para afirmar causalidade do encontro nem
  provar que o gosto antecede a amizade (amigos se influenciam). Declarar no relatório.
- As features do encontro (local, duração, RSSI) não carregam sinal real do rótulo.
- Licença: uso não comercial; citar Last.fm e o workshop HetRec 2011 (Cantador et al.).
