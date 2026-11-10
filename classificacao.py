"""
Parte 1: Classificação de Diagnóstico de Câncer de Mama
Dataset: Breast Cancer Wisconsin (Scikit-Learn)
Modelo: Random Forest Classifier
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

def executar_classificacao():
    print("=" * 70)
    print("PARTE 1: CLASSIFICAÇÃO - BREAST CANCER WISCONSIN")
    print("=" * 70)

    # 1. Carregamento dos dados
    raw_data = load_breast_cancer()
    df = pd.DataFrame(data=raw_data.data, columns=raw_data.feature_names)
    df['target'] = raw_data.target  # 0: Maligno, 1: Benigno

    # Mapeamento legível das classes
    target_names = {0: 'Maligno', 1: 'Benigno'}
    df['classe_nome'] = df['target'].map(target_names)

    # 2. Análise Exploratória Simples (EDA)
    print("\n[EDA] Dimensões do Dataset:")
    print(f"- Total de Registros: {df.shape[0]}")
    print(f"- Quantidade de Atributos: {len(raw_data.feature_names)}")

    print("\n[EDA] Distribuição das Classes:")
    dist_classes = df['classe_nome'].value_counts()
    for classe, count in dist_classes.items():
        pct = (count / len(df)) * 100
        print(f"  * {classe}: {count} instâncias ({pct:.2f}%)")

    print("\n[EDA] Estatísticas Descritivas (5 primeiros atributos):")
    print(df[raw_data.feature_names[:5]].describe().round(3))

    # 3. Separação de X e y
    X = df[raw_data.feature_names]
    y = df['target']

    # 4. Divisão Treino e Teste (Estratificada)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\n[Split] Amostras Treino: {X_treino.shape[0]} | Amostras Teste: {X_teste.shape[0]}")

    # 5. Treinamento do Modelo (Random Forest)
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    modelo_rf.fit(X_treino, y_treino)

    # Predições
    y_pred = modelo_rf.predict(X_teste)

    # 6. Avaliação do Desempenho
    acc = accuracy_score(y_teste, y_pred)
    prec_mal = precision_score(y_teste, y_pred, pos_label=0)  # Classe 0 = Maligno
    rec_mal = recall_score(y_teste, y_pred, pos_label=0)
    f1_mal = f1_score(y_teste, y_pred, pos_label=0)
    cm = confusion_matrix(y_teste, y_pred)

    print("\n" + "-" * 55)
    print("MÉTRICAS DE AVALIAÇÃO (Foco na Classe Crítica: Maligno = 0)")
    print("-" * 55)
    print(f"Acurácia Global : {acc * 100:.2f}%")
    print(f"Precisão        : {prec_mal * 100:.2f}%")
    print(f"Recall (Sensib.): {rec_mal * 100:.2f}%")
    print(f"F1-Score        : {f1_mal * 100:.2f}%")

    print("\nRelatório de Classificação Completo:")
    print(classification_report(y_teste, y_pred, target_names=['Maligno', 'Benigno']))

    # Matriz de Confusão
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Maligno (Pred)', 'Benigno (Pred)'],
        yticklabels=['Maligno (Real)', 'Benigno (Real)']
    )
    plt.title('Matriz de Confusão - Random Forest (Câncer de Mama)')
    plt.tight_layout()
    plt.show()

    # 7. Breve Análise dos Resultados
    print("\n[ANÁLISE DOS RESULTADOS - PARTE 1]")
    print(f"O classificador Random Forest obteve acurácia de {acc*100:.2f}%. Na triagem médica,")
    print("a métrica mais crítica é o Recall para a classe Maligno, visto que classificar um")
    print("tumor maligno como benigno (falso negativo) acarreta atraso no tratamento do paciente.")
    print(f"O modelo alcançou Recall de {rec_mal*100:.2f}%, demonstrando alta sensibilidade diagnóstica.")

if __name__ == '__main__':
    executar_classificacao()