# Comparação de modelos — OpiniOn (CC4)

Gerado por `python -m opinion.modelo.comparar` em 27/11/2026 19:41. Números do teste medidos **uma vez** por modelo; nada foi escolhido com eles.

> Usuários, obras e notas são reais (Amazon Reviews 2023); encontros e o label `conectou` são simulados ([`fontes_dados.md`](fontes_dados.md)). As métricas medem quão bem cada modelo recupera o processo da simulação, não o comportamento de usuários reais da pulseira.

## 1. Protocolo

- **Mesmos dados e split do CC3:** treino semanas 1–8 (46.244 encontros), validação 9–10, teste 11–12; purga na fronteira treino/validação.
- **Cada família** tem uma grade pequena e fixa (`config/modelo.yaml → comparacao`). A configuração e o limiar do sinal (F0,5 entre desconhecidos) são escolhidos na validação. Otimização de hiperparâmetros fica para o CC5 (Optuna) e validação cruzada para o CC6.
- **Escolha do melhor modelo:** pr_auc_desconhecidos na validação. A regra simples (`compat_score`) recebe o mesmo protocolo, com limiar próprio.
- **Incerteza:** intervalo de confiança de 95% por bootstrap pareado (8 pontuações, mesmas reamostras para todas) no teste, entre desconhecidos.

## 2. Resultado

| Modelo | PR-AUC desc. (valid.) | PR-AUC desc. (teste, IC 95%) | PR-AUC geral (teste) | ROC-AUC desc. (teste) | Brier (teste) |
|---|---|---|---|---|---|
| AdaBoost **(escolhido)** | 0,136 | 0,186 (0,148–0,246) | 0,269 | 0,955 | 0,088 |
| Gradient Boosting | 0,130 | 0,176 (0,139–0,228) | 0,274 | 0,953 | 0,061 |
| Random Forest | 0,128 | 0,184 (0,145–0,239) | 0,265 | 0,949 | 0,060 |
| Árvore de decisão | 0,116 | 0,156 (0,125–0,195) | 0,255 | 0,954 | 0,066 |
| MLP (rede neural) | 0,107 | 0,139 (0,106–0,189) | 0,264 | 0,928 | 0,062 |
| Regressão logística | 0,101 | 0,159 (0,120–0,223) | 0,266 | 0,927 | 0,147 |
| KNN | 0,060 | 0,101 (0,073–0,153) | 0,265 | 0,876 | 0,061 |
| Regra simples | 0,100 | 0,131 (0,104–0,177) | 0,103 | 0,937 | não se aplica |
| *acaso* | | *0,014* | *0,080* | *0,500* | *0,073* (constante) |

![PR-AUC com IC](img/cc4/pr_auc_ic.png)

![Curvas PR](img/cc4/curvas_pr.png)

![Curvas ROC](img/cc4/curvas_roc.png)

## 3. No ponto de operação (desconhecidos, teste)

O limiar do modelo é uma avaliação acadêmica. O aviso do protótipo usa o corte por categoria na proximidade.

| Modelo | Limiar | Precisão | Recall | F0,5 | Sinal em | VP | FP | FN | VN |
|---|---|---|---|---|---|---|---|---|---|
| AdaBoost | 0,313 | 24,1% | 11,4% | 0,197 | 0,7% | 14 | 44 | 109 | 8.666 |
| Gradient Boosting | 0,123 | 18,6% | 48,0% | 0,211 | 3,6% | 59 | 259 | 64 | 8.451 |
| Random Forest | 0,134 | 21,8% | 30,9% | 0,232 | 2,0% | 38 | 136 | 85 | 8.574 |
| Árvore de decisão | 0,163 | 13,1% | 13,0% | 0,131 | 1,4% | 16 | 106 | 107 | 8.604 |
| MLP (rede neural) | 0,054 | 12,7% | 51,2% | 0,150 | 5,6% | 63 | 433 | 60 | 8.277 |
| Regressão logística | 0,416 | 19,7% | 12,2% | 0,176 | 0,9% | 15 | 61 | 108 | 8.649 |
| KNN | 0,061 | 11,4% | 17,9% | 0,123 | 2,2% | 22 | 171 | 101 | 8.539 |
| Regra simples | 0,773 | 18,8% | 9,8% | 0,158 | 0,7% | 12 | 52 | 111 | 8.658 |

![Matrizes de confusão](img/cc4/matrizes_confusao.png)

## 4. Diferenças e incerteza (PR-AUC entre desconhecidos, teste)

| Modelo | vs Regressão logística (IC 95%) | vs regra simples (IC 95%) |
|---|---|---|
| AdaBoost | +0,027 (-0,024 a +0,074) | +0,055 (+0,015 a +0,099) |
| Gradient Boosting | +0,017 (-0,040 a +0,065) | +0,045 (+0,003 a +0,092) |
| Random Forest | +0,025 (-0,029 a +0,077) | +0,053 (+0,010 a +0,101) |
| Árvore de decisão | -0,003 (-0,060 a +0,038) | +0,025 (-0,014 a +0,058) |
| MLP (rede neural) | -0,020 (-0,068 a +0,019) | +0,008 (-0,023 a +0,040) |
| Regressão logística | — | +0,028 (+0,003 a +0,061) |
| KNN | -0,058 (-0,090 a -0,027) | -0,030 (-0,056 a +0,005) |

## 5. Precisão do sinal por contexto (desconhecidos, teste)

| Local | Encontros | Taxa real | AdaBoost | Gradient Boosting | Random Forest | Árvore de decisão | MLP (rede neural) | Regressão logística | KNN | Regra simples |
|---|---|---|---|---|---|---|---|---|---|---|
| campus | 1.716 | 0,8% | 40,0% (0,3%) | 14,3% (0,4%) | 0,0% (0,0%) | 0,0% (0,1%) | 10,8% (2,2%) | 25,0% (0,7%) | 13,0% (1,3%) | 27,3% (0,6%) |
| cinema | 1.561 | 1,6% | 33,3% (1,0%) | 19,3% (5,6%) | 25,4% (4,3%) | 11,1% (2,3%) | 13,6% (7,0%) | 16,7% (0,8%) | 10,7% (3,6%) | 16,7% (0,8%) |
| convencao_games | 1.909 | 2,3% | 21,9% (1,7%) | 23,2% (7,2%) | 20,9% (4,8%) | 17,9% (2,9%) | 16,5% (8,9%) | 25,8% (1,6%) | 17,5% (3,0%) | 22,7% (1,2%) |
| livraria | 3.647 | 1,2% | 0,0% (0,2%) | 10,6% (2,3%) | 12,5% (0,4%) | 6,9% (0,8%) | 8,9% (4,9%) | 9,5% (0,6%) | 5,3% (1,6%) | 10,5% (0,5%) |

Cada célula: precisão (fração de encontros com sinal).

## 6. O que cada modelo usa (importância por permutação, validação)

Queda de PR-AUC entre desconhecidos quando a feature é embaralhada. Top 5 de cada modelo:

| Modelo | 1º | 2º | 3º | 4º | 5º |
|---|---|---|---|---|---|
| AdaBoost | `compat_score` (+0,119) | `log_duracao` (+0,039) | `compat_score_filme` (+0,015) | `compat_score_jogo` (+0,007) | `compat_score_musica` (+0,007) |
| Gradient Boosting | `compat_score` (+0,118) | `historico_min` (+0,014) | `compat_score_livro` (+0,004) | `conexoes_min` (+0,004) | `sim_generos` (+0,003) |
| Random Forest | `compat_score` (+0,043) | `sim_generos` (+0,021) | `compat_score_filme` (+0,008) | `local_livraria` (+0,004) | `hora` (+0,003) |
| Árvore de decisão | `compat_score` (+0,106) | `log_duracao` (+0,017) | `hora` (+0,007) | `conexoes_min` (+0,006) | `local_campus` (+0,002) |
| MLP (rede neural) | `sim_generos` (+0,094) | `match_filme` (+0,013) | `historico_max` (+0,010) | `compat_score` (+0,010) | `local_campus` (+0,010) |
| Regressão logística | `compat_score_livro` (+0,086) | `compat_score_musica` (+0,065) | `sim_generos` (+0,064) | `match_serie` (+0,045) | `compat_score_serie` (+0,044) |
| KNN | `sim_generos` (+0,019) | `compat_score_jogo` (+0,012) | `match_jogo` (+0,011) | `compat_score_livro` (+0,006) | `sim_notas` (+0,005) |

![Importância](img/cc4/importancia.png)

## 7. Custo

| Modelo | Configurações testadas | Treino por configuração | Pontuação por encontro |
|---|---|---|---|
| AdaBoost | 4 | 14,0 s | 0,018 ms |
| Gradient Boosting | 4 | 0,7 s | 0,004 ms |
| Random Forest | 4 | 5,0 s | 0,014 ms |
| Árvore de decisão | 6 | 0,2 s | 0,002 ms |
| MLP (rede neural) | 4 | 3,3 s | 0,004 ms |
| Regressão logística | 6 | 0,3 s | 0,002 ms |
| KNN | 4 | 1,4 s | 0,150 ms |

## 8. Análise crítica

- **Escolhido na validação: AdaBoost** (0,136). No teste a ordem muda (1º no teste: AdaBoost): as diferenças entre eles são pequenas o bastante para mudar de posição com outra amostra.
- **Contra o baseline (Regressão logística):** +0,027 de PR-AUC (IC 95% -0,024 a +0,074). O intervalo inclui zero: **não dá para afirmar que o modelo mais complexo é melhor que o baseline** com estes dados.
- **Contra a regra simples:** +0,055 de PR-AUC (IC 95% +0,015 a +0,099) — o modelo supera a regra de forma consistente.
- **No ponto de operação**, AdaBoost acerta 24,1% dos sinais (recall 11,4%, sinal em 0,7% dos encontros); a regra, 18,8% (recall 9,8%, sinal em 0,7%). A vantagem aparece também na precisão do sinal, que é o que o usuário sente.
- **Calibração:** prever sempre a prevalência dá Brier 0,073. Regressão logística e AdaBoost ficam pior que isso (probabilidades descalibradas: `class_weight=balanced` infla as da regressão logística e o AdaBoost concentra as suas perto de 0,5); Árvore de decisão, Random Forest, Gradient Boosting, KNN e MLP (rede neural) ficam melhor. A ordenação e o limiar não dependem disso, mas a probabilidade só pode ser mostrada como chance real (dashboard, faixas de sinal) depois de calibrada.
- **O que os modelos aprendem:** as 3 features mais importantes variam entre os modelos. Lembrete: a importância reflete as premissas do simulador, não uma descoberta sobre usuários.
- **Margem:** o teto do simulador (a probabilidade escondida) chega a PR-AUC 0,396 e precisão 53,4% com a mesma taxa de sinal; AdaBoost está em 0,186 e 24,1%. A parte que falta vem do que nenhum dado observa (a abertura social de cada pessoa) e do sorteio do resultado.
- **Custo:** AdaBoost pontua um encontro em 0,018 ms, sem impedimento para o sinal em tempo real (CC7).
- **Recomendação para o CC5:** otimizar AdaBoost com Optuna, mantendo o baseline como referência explicável.

## 9. Limitações

- **Label simulado.** O ranking dos modelos vale para o processo da simulação. Com outras premissas (ver os cenários em [`validacao_simulacao.md`](validacao_simulacao.md)) a ordem entre modelos próximos pode mudar.
- **Um único split temporal.** O IC do bootstrap cobre a variação de amostragem do teste, não a de treinar em outras semanas. Estabilidade entre folds é o CC6.
- **Grades fixas e pequenas.** Um modelo pode ter perdido só por falta de ajuste; o CC5 otimiza o vencedor com Optuna.
- **Teto do simulador.** Só existe porque os dados são simulados; serve para dimensionar a margem, não como meta realista.
