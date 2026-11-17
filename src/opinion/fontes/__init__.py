"""
Fontes de dados reais (Frente A, Sprint 3).

Cada fonte baixa os arquivos públicos para data/externo/ (fora do Git) e gera, em
data/reais/<fonte>/, tabelas no mesmo formato do histórico do app (usuarios, obras,
avaliacoes). O simulador usa essas tabelas no lugar das inventadas
(config/simulacao.yaml -> usuarios.fonte).
"""
from opinion.mundo import RAIZ

PASTA_EXTERNO = RAIZ / "data" / "externo"
PASTA_REAIS = RAIZ / "data" / "reais"
CONFIG_FONTES = RAIZ / "config" / "fontes.yaml"
