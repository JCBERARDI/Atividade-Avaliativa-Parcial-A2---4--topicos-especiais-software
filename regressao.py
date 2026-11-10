"""
Parte 2: Regressão Linear para Previsão de Preços de Imóveis
Dataset: California Housing (Scikit-Learn)
Modelo: Linear Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def executar_regressao():
    print("=" * 70)
    print("PARTE 2: REGRESSÃO LINEAR - PREVISÃO DE PREÇOS (CALIFORNIA HOUSING)")
    print("=" * 70)

    # 1. Carregamento dos dados
    raw_housing = fetch_california_housing()
    df_housing = pd.DataFrame(data=raw_housing.data, columns=raw_housing.feature_names)
    # A variável alvo está em unidades de centenas de milhares de dólares ($100.000)
    df_housing['MedHouseVal'] = raw_housing.target

    # 2. Análise Exploratória (EDA)
    print("\n[EDA] Dimensões do Dataset:")
    print(f"- Total de Registros: {df_housing.shape[0]}")
    print(f"- Quantidade de Atributos Preditores: {len(raw_housing.feature_names)}")

    print("\n[EDA] Resumo Estatístico dos Atributos Principais:")
    print(df_housing[['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal']].describe().round(3))

    nulos = df_housing.isnull().sum().sum()
    print(f"\n[EDA] Total de Valores Nulos: {nulos}")

    # 3. Identificação das Variáveis
    X = df_housing[raw_housing.feature_names]
    y = df_housing['MedHouseVal']

    # 4. Divisão Treino e Teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[Split] Amostras Treino: {X_treino.shape[0]} | Amostras Teste: {X_teste.shape[0]}")

    # Normalização dos preditores
    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)

    # 5. Treinamento da Regressão Linear
    reg_linear = LinearRegression()
    reg_linear.fit(X_treino_scaled, y_treino)

    # Predições
    y_pred = reg_linear.predict(X_teste_scaled)

    # 6. Avaliação do Desempenho
    mae = mean_absolute_error(y_teste, y_pred)
    mse = mean_squared_error(y_teste, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_teste, y_pred)

    print("\n" + "-" * 55)
    print("MÉTRICAS DE AVALIAÇÃO DA REGRESSÃO LINEAR")
    print("-" * 55)
    print(f"MAE  (Erro Médio Absoluto)      : {mae:.4f} (~ ${mae * 100000:,.2f})")
    print(f"MSE  (Erro Quadrático Médio)     : {mse:.4f}")
    print(f"RMSE (Raiz do Erro Quadr. Médio) : {rmse:.4f} (~ ${rmse * 100000:,.2f})")
    print(f"R²   (Coeficiente de Determinação): {r2:.4f}")

    # 7. Gráfico: Valores Reais vs. Previstos
    plt.figure(figsize=(7, 6))
    plt.scatter(y_teste, y_pred, alpha=0.25, color='#1f77b4', edgecolors='none', s=20)
    limites = [min(y_teste.min(), y_pred.min()), max(y_teste.max(), y_pred.max())]
    plt.plot(limites, limites, color='red', linestyle='--', linewidth=2, label='Linha Ideal (y_real = y_pred)')

    plt.title('Valores Reais vs. Valores Previstos - Regressão Linear')
    plt.xlabel('Preço Real ($100k)')
    plt.ylabel('Preço Previsto ($100k)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 8. Interpretação dos Resultados
    print("\n[INTERPRETAÇÃO DOS RESULTADOS - PARTE 2]")
    print(f"O modelo linear explicou {r2*100:.2f}% da variabilidade do preço dos imóveis (R² = {r2:.4f}).")
    print(f"O MAE de {mae:.3f} indica um desvio absoluto médio de ${mae*100000:,.2f} por residência.")
    print("O gráfico expõe um limite artificial de corte em 5.0 ($500.000) nos dados reais,")
    print("o que afeta a precisão da reta linear para propriedades de maior valor.")

if __name__ == '__main__':
    executar_regressao()