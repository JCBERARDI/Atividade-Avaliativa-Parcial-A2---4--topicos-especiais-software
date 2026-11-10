"""
Orquestrador Geral da Atividade Avaliativa de Machine Learning.
Executa os fluxos de Classificação e Regressão, imprimindo a conclusão final.
"""

from classificacao import executar_classificacao
from regressao import executar_regressao

def imprimir_conclusao():
    print("\n" + "=" * 70)
    print("8. CONCLUSÃO CRÍTICA INTEGRADA")
    print("=" * 70)
    texto = """
1. Classificação (Diagnóstico Oncológico):
   - O Random Forest demonstrou alta capacidade discriminativa com dados tabulares
     multidimensionais sem necessidade de seleção manual de atributos.
   - O trade-off entre Precisão e Recall foi priorizado em favor do Recall na classe
     Maligno, reduzindo drasticamente falsos negativos, aspecto fulcral em saúde pública.

2. Regressão Linear (Mercado Imobiliário):
   - A Regressão Linear estabeleceu uma linha de base interpretável (R² em torno de 0.60),
     confirmando a renda mediana local (MedInc) como o preditor de maior peso.
   - Como limitação estrutural, o modelo linear não captura plenamente a não-linearidade
     espacial (coordenadas geográficas) e sofre com a censura superior de preços (teto de 5.0),
     onde modelos baseados em gradiente (ex: XGBoost ou LightGBM) seriam mais indicados.
"""
    print(texto)
    print("=" * 70)

if __name__ == '__main__':
    executar_classificacao()
    print("\n\n")
    executar_regressao()
    imprimir_conclusao()