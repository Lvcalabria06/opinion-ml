# Match por mídia nos encontros do OpiniOn

## Aviso na proximidade

A avaliação liga uma pessoa a uma obra por `usuario_id` e `obra_id`. O catálogo identifica a mídia
e seus gêneros. Para cada participante, somamos `nota − 3` por gênero dentro da mídia, usando
as notas anteriores ao início do encontro. Duas pessoas que avaliaram discos diferentes podem
combinar por gêneros preferidos em comum e similaridade positiva. Uma obra favorita de ambos
também serve de evidência. A concordância das notas pesa quando há obras avaliadas pelos dois.

O corte inicial de afinidade alta é **0,68**, configurável em `config/pipeline.yaml`.
Cada mídia precisa atingir esse valor sozinha. Ao detectar proximidade, a pulseira pode
receber um único aviso com vários assuntos. O texto preserva o anonimato da outra pessoa.
Abertura do perfil, follow e demais interações elegíveis contam no desfecho de 48 horas
do encontro. Um encontro com duas mídias sinalizadas tem um único desfecho observado.

### Verificação com o bootstrap atual

Dos 69.384 encontros simulados, 6.972 atingiram o corte em alguma mídia, 815 em música
e 209 em duas ou mais mídias. Os motivos calculados pelo pipeline coincidiram com os
registrados pelo simulador em todos os encontros. Entre desconhecidos no teste temporal,
houve interação em 17 de 122 encontros com aviso musical e em 121 de 970 encontros
com algum aviso.

No modelo de música, Random Forest foi escolhido na validação. No teste, a PR-AUC foi
0,264 no recorte completo e 0,200 entre desconhecidos. A prevalência entre desconhecidos
foi 0,023 e a regra de afinidade alcançou PR-AUC de 0,102 nesse grupo. Esses valores
avaliam todos os pares com histórico musical bilateral, incluindo os que ficaram abaixo
do corte do aviso. O simulador condiciona as conexões geradas pelo encontro à presença
do aviso e mantém interações espontâneas do feed e entre amigos. O efeito do aviso em
pessoas reais exige observação independente.

## Experimento histórico com os rótulos anteriores

Em 18/11/2026, o pipeline produziu 65.443 encontros rotulados para treino,
validação e teste. Em 40.008 deles houve afinidade em pelo menos uma mídia,
e em 12.461 houve afinidade em duas ou mais. No encontro `e-000015`, o par
`usr-00327|usr-00777` teve score 0,4314 em filme e 0,5925 em música
sem ter avaliado uma mesma obra. Os scores expressam compatibilidade de gosto.
O modelo estima a probabilidade de interação nas 48 horas seguintes.

## Como foi medido

Usamos o bootstrap versionado para manter constantes os participantes, encontros,
rótulos e splits. A mudança de elegibilidade de duas mídias para uma se aplica à
próxima extração da Amazon. A amostra desta comparação continua com 1.200 usuários.
Cada recorte contém todos os encontros com histórico bilateral naquela mídia, inclusive
quando os gostos divergem. Um encontro que aparece em dois experimentos continua no
mesmo split temporal. Os números de encontros dos recortes se sobrepõem.

Em cada recorte foram treinadas Regressão Logística, Árvore de Decisão, Random Forest
e Gradient Boosting com as grades existentes. A seleção usa PR-AUC na validação
entre desconhecidos. A referência abaixo é o Gradient Boosting da abordagem original,
retreinado sobre o conjunto geral e avaliado **nos mesmos encontros de teste de cada linha**.

| População | Encontros de treino | Desconhecidos no teste | Selecionado na validação | PR-AUC no teste | Referência original no mesmo recorte |
|---|---:|---:|---|---:|---:|
| Todos os encontros | 44.254 | 8.639 | Gradient Boosting | 0,143 | 0,198 |
| Filmes | 31.725 | 6.135 | Random Forest | 0,137 | 0,210 |
| Música | 5.625 | 1.133 | Random Forest | 0,134 | 0,204 |
| Séries | 6.748 | 1.396 | Random Forest | 0,111 | 0,185 |
| Jogos | 2.395 | 565 | Gradient Boosting | 0,228 | 0,290 |
| Livros | 28.402 | 5.518 | Random Forest | 0,123 | 0,179 |

As estimativas pontuais caíram nos cinco recortes. O gerador de conexões usa uma
afinidade latente calculada sobre um perfil agregado de gêneros. Isso favorece, em
princípio, informações que refletem esse perfil agregado. O experimento mede a troca
da representação de gosto e a restrição da amostra com esse mecanismo de geração de
rótulos. O efeito de acrescentar novas populações ainda precisa ser medido.

Para reproduzir uma linha, siga o [README](../README.md) e execute o comparador
com a mídia correspondente. O experimento no MLflow contém os quatro modelos,
a regra simples, os limiares, as matrizes de confusão e os intervalos de confiança.
