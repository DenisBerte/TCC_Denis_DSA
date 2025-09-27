# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 17:37:15 2025

@author: denis
"""

#%%  funções de ajuda 

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, balanced_accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def descritiva(df_, var, vresp='DS_SIT_TOT_TURNO', max_classes=5):
    """
    Gera um gráfico bivariado: taxa do evento (vresp) por categoria de `var`
    + barras de frequência no eixo secundário.
    Mantém mudanças mínimas, mas:
      - vresp default = 'DS_SIT_TOT_TURNO'
      - evita qcut para dummies (0/1)
      - mapeia texto 'ELEITO'/'NÃO ELEITO' para 1/0 se necessário
    """
    df = df_.copy()

    # garante alvo binário 0/1 se vier como texto
    if df[vresp].dtype == object:
        df[vresp] = df[vresp].map({'NÃO ELEITO': 0, 'ELEITO': 1})
    df[vresp] = df[vresp].astype(float)

    # aplica qcut só se for numérica contínua com muitas classes e não for dummy
    x = df[var]
    if np.issubdtype(x.dtype, np.number):
        uniq = np.unique(x[~pd.isna(x)])
        is_dummy = set(pd.Series(x).dropna().unique()).issubset({0, 1})
        if (len(uniq) > max_classes) and (not is_dummy):
            df[var] = pd.qcut(x, max_classes, duplicates='drop')
    else:
        df[var] = df[var].astype('category')

    # plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.pointplot(data=df, y=vresp, x=var, ax=ax1, errorbar='ci')
    ax1.set_ylabel('Taxa do evento')

    ax2 = ax1.twinx()
    sns.countplot(data=df, x=var, palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax1.set_zorder(2)
    ax1.patch.set_visible(False)
    plt.tight_layout()
    plt.show()


def avalia_clf(clf, y, X, rótulos_y=['Não Eleito', 'Eleito'], base='treino'):
    """
    Avalia classificador: acurácia, acurácia balanceada, AUC/GINI, matriz de confusão,
    relatório de classificação e curva ROC. Mudança mínima: rótulos padrão atualizados.
    """
    # Predições
    pred = clf.predict(X)

    # Probabilidade do evento (classe positiva)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X)[:, -1]
    elif hasattr(clf, "decision_function"):
        # fallback para modelos sem predict_proba
        # normaliza para [0,1] de forma monotônica
        s = clf.decision_function(X)
        y_prob = (s - s.min()) / (s.max() - s.min() + 1e-12)
    else:
        # último recurso: usa as predições como "probabilidade"
        y_prob = pred.astype(float)

    # Métricas básicas
    cm = confusion_matrix(y, pred)
    ac = accuracy_score(y, pred)
    bac = balanced_accuracy_score(y, pred)

    print(f'\nBase de {base}:')
    print(f'Acurácia: {ac:.1%}')
    print(f'Acurácia balanceada: {bac:.1%}')

    # AUC / GINI
    try:
        auc_score = roc_auc_score(y, y_prob)
        print(f"AUC-ROC: {auc_score:.2%}")
        print(f"GINI: {(2*auc_score-1):.2%}")
    except Exception as e:
        auc_score = np.nan
        print(f"AUC-ROC: não calculada ({e})")

    # Matriz de confusão
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=rótulos_y, yticklabels=rótulos_y)
    plt.title(f"Matriz de confusão - base de {base}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    plt.show()

    # Relatório de classificação
    print('\n', classification_report(y, pred, target_names=rótulos_y, zero_division=0))

    # Curva ROC
    if not np.isnan(auc_score):
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"Curva ROC - base de {base}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()


def relatorio_missing(df):
    """
    Mantida caso você queira usar. Retorna resumo de missing.
    """
    print(f'Número de linhas: {df.shape[0]} | Número de colunas: {df.shape[1]}')
    return pd.DataFrame({
        'Pct_missing': df.isna().mean().apply(lambda x: f"{x:.1%}"),
        'Freq_missing': df.isna().sum().apply(lambda x: f"{x:,.0f}").replace(',', '.')
    })


def diagnóstico(df_, var, vresp='DS_SIT_TOT_TURNO', pred='pred', max_classes=5):
    """
    Compara taxa observada (vresp) vs predita (pred) por categorias de `var`,
    com barras de frequência no eixo secundário. Mudanças mínimas:
      - vresp default = 'DS_SIT_TOT_TURNO'
      - evita qcut para dummies 0/1
      - mapeia texto 'ELEITO'/'NÃO ELEITO' se necessário
    """
    df = df_.copy()

    # garante alvo binário 0/1 se vier como texto
    if df[vresp].dtype == object:
        df[vresp] = df[vresp].map({'NÃO ELEITO': 0, 'ELEITO': 1})
    df[vresp] = df[vresp].astype(float)

    # prepara variável (qcut só quando fizer sentido)
    df[var + '_'] = df[var]
    x = df[var + '_']
    if np.issubdtype(x.dtype, np.number):
        uniq = np.unique(x[~pd.isna(x)])
        is_dummy = set(pd.Series(x).dropna().unique()).issubset({0, 1})
        if (len(uniq) > max_classes) and (not is_dummy):
            df[var + '_'] = pd.qcut(x, max_classes, duplicates='drop')
    else:
        df[var + '_'] = df[var + '_'].astype('category')

    # plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.pointplot(data=df, y=vresp, x=var + '_', ax=ax1, errorbar='ci', label='Observado')
    sns.pointplot(data=df, y=pred,  x=var + '_', ax=ax1, color='red',
                  linestyles='--', errorbar=None, label='Predito')
    ax1.legend(loc='best')
    ax1.set_ylabel('Taxa')

    ax2 = ax1.twinx()
    sns.countplot(data=df, x=var + '_', palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax1.set_zorder(2)
    ax1.patch.set_visible(False)
    plt.tight_layout()
    plt.show()
