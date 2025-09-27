# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 15:28:10 2025

@author: denis
"""

#%% biblios

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler #balanceamento


#%% base geral

dfdep = pd.read_excel("dfdep.xlsx")

dfdep.info()

#%%bases para arvore - variaveis

# Lista de variáveis para as analises de rf e regressao

variaveis_rf = [
    "DS_SIT_TOT_TURNO",
    "DS_GENERO",
    "DS_GRAU_INSTRUCAO",
    "DS_ESTADO_CIVIL",
    "DS_COR_RACA",
    "RELIGIOSOS",
    "SEGURANCA",
    "ESPECTRO_POL",
    "FAIXA_ETARIA",
    "ANO_ELEICAO",
    "REGIAO"
]
#%% criar base separada - - base de estaduais
#base só com candidatos a deputados estaduais
# são dados de 3 eleições

# Filtra dados religiosos
dfdep_est = dfdep[(dfdep["DS_CARGO"] == "DEPUTADO ESTADUAL")][variaveis_rf]


# Salva o dataframe em um arquivo separado
dfdep_est.to_excel("dfdep_est.xlsx", index=False)


#%% ########################
# RANDOM FOREST - manipulações base Religiosos

#%% abrir a base

dfdep_est = pd.read_excel("dfdep_est.xlsx")

dfdep_est.info()


#%%
# Contar quantos têm RELIGIOSOS = 1
conta_religiosos = dfdep_est[dfdep_est["RELIGIOSOS"] == 1].shape[0]

# Contar quantos têm SEGURANCA = 1
conta_seguranca = dfdep_est[dfdep_est["SEGURANCA"] == 1].shape[0]

# Exibir resultados
print("RELIGIOSOS = 1:", conta_religiosos)
print("SEGURANCA = 1:", conta_seguranca)

#%% RELIGIOSOS e SEGURANCA eleitos por ano + total combinado

# RELIGIOSOS eleitos por ano
religiosos_eleitos_ano = dfdep_est[
    (dfdep_est["RELIGIOSOS"] == 1) &
    (dfdep_est["DS_SIT_TOT_TURNO"] == "ELEITO")
].groupby("ANO_ELEICAO").size()

# SEGURANCA eleitos por ano
seguranca_eleitos_ano = dfdep_est[
    (dfdep_est["SEGURANCA"] == 1) &
    (dfdep_est["DS_SIT_TOT_TURNO"] == "ELEITO")
].groupby("ANO_ELEICAO").size()

# tabela
df_eleitos_por_ano = pd.DataFrame({
    "RELIGIOSOS_ELEITOS": religiosos_eleitos_ano,
    "SEGURANCA_ELEITOS": seguranca_eleitos_ano
}).fillna(0).astype(int)

# Total combinado (soma simples)
df_eleitos_por_ano["TOTAL_CATEGORIAS"] = (
    df_eleitos_por_ano["RELIGIOSOS_ELEITOS"] +
    df_eleitos_por_ano["SEGURANCA_ELEITOS"]
)

# linha de totais gerais
df_eleitos_por_ano.loc["TOTAL"] = df_eleitos_por_ano.sum()

print(df_eleitos_por_ano)

# Salva a tabela em Excel
df_eleitos_por_ano.to_excel("n_eleitos_estaduais.xlsx", index=True)


#%% Manipulações

# Garantir que 'DS_SIT_TOT_TURNO' esteja binária: 1 = ELEITO, 0 = NÃO ELEITO
dfdep_est['DS_SIT_TOT_TURNO'] = (
    dfdep_est['DS_SIT_TOT_TURNO']
    .map({'NÃO ELEITO': 0, 'ELEITO': 1})
    .fillna(0)
    .astype(int)
)

# DS_GENERO: 1 = MASCULINO, 0 = FEMININO (ref: feminino)
dfdep_est['DS_GENERO'] = (
    dfdep_est['DS_GENERO']
    .map({'FEMININO': 0, 'MASCULINO': 1})
    .fillna(0)
    .astype(int)
)

# DS_COR_RACA: 1 = BRANCA, 0 = NÃO BRANCA (ref: não branca)
dfdep_est['DS_COR_RACA'] = (
    dfdep_est['DS_COR_RACA']
    .map({'NÃO BRANCA': 0, 'BRANCA': 1})
    .fillna(0)
    .astype(int)
)

#%% DS_GRAU_INSTRUCAO: 1='Até EF', 2='Até EM', 3='Até Sup.'
dfdep_est['DS_GRAU_INSTRUCAO'] = (
    dfdep_est['DS_GRAU_INSTRUCAO']
    .map({'Até EF': 1, 'Até EM': 2, 'Até Sup.': 3})
    .fillna(0)
    .astype(int)
)

# Dummies com drop da primeira categoria (ref: Até EF)
dfdep_est = pd.get_dummies(dfdep_est, columns=['DS_GRAU_INSTRUCAO'], drop_first=True)

#%% FAIXA_ETARIA: 1='18–29', 2='30–44', 3='45–59', 4='60+'
dfdep_est['FAIXA_ETARIA'] = (
    dfdep_est['FAIXA_ETARIA']
    .map({'18–29': 1, '30–44': 2, '45–59': 3, '60+': 4})
    .fillna(0)
    .astype(int)
)

# Dummies com drop da primeira categoria (referência: 18–29 anos)
dfdep_est = pd.get_dummies(dfdep_est, columns=['FAIXA_ETARIA'], drop_first=True)


#%% # Estado civil (referência: SOLTEIRO(A))

dfdep_est = pd.get_dummies(dfdep_est, columns=['DS_ESTADO_CIVIL'], drop_first=False)
dfdep_est = dfdep_est.drop(columns=['DS_ESTADO_CIVIL_SOLTEIRO(A)'])

# Espectro político (referência: Centro)
dfdep_est = pd.get_dummies(dfdep_est, columns=['ESPECTRO_POL'], drop_first=True)


# Grande região - (referência: Sudeste)
dfdep_est = pd.get_dummies(dfdep_est, columns=['REGIAO'], drop_first=False)
dfdep_est = dfdep_est.drop(columns=['REGIAO_SUDESTE'])


#%% # Converter todas as colunas booleanas para inteiras
for col in dfdep_est.columns:
    if dfdep_est[col].dtype == bool:
        dfdep_est[col] = dfdep_est[col].astype(int)


#%%
print(dfdep_est.dtypes)
print(dfdep_est.head())

#%% salvar original e copia para analises

dfdep_est.to_excel("dfdep_est.xlsx", index=False)


# Cópia: Atribuir a base final manipulada
df_estad = dfdep_est.copy()

# Salvar para analises
df_estad.to_excel("df_estad.xlsx", index=False)


#%%######################### Random Forest ####################################
###############################################################################
#%% abrir a base (começo do código)

df_estad = pd.read_excel("df_estad.xlsx")

print(df_estad.head())
print(df_estad.info())


#%% descritiva bivariada

from funcoes_ajuda import descritiva

#Definindo a lista de features para descritiva
variaveis = list(df_estad.columns)
vResp = 'DS_SIT_TOT_TURNO'

print(variaveis)
print(vResp)

#%% análise descritiva bivariada
for var in variaveis:
    descritiva(df_estad, var, vResp, 2)

#%% descritiva univariada

# Dicionário de rótulos por variável 
mapas_categorias = {
    'DS_SIT_TOT_TURNO': {0: 'não eleito', 1: 'eleito'},
    'DS_GENERO': {0: 'feminino', 1: 'masculino'},
    'DS_COR_RACA': {0: 'não branca', 1: 'branca'},
    'RELIGIOSOS': {0: 'não', 1: 'sim'},
    'SEGURANCA': {0: 'não', 1: 'sim'},
    'DS_GRAU_INSTRUCAO_2': {0: 'não EM', 1: 'EM'},
    'DS_GRAU_INSTRUCAO_3': {0: 'não ES', 1: 'ES'},
    'FAIXA_ETARIA_2': {0: 'não 30-44', 1: '30-44'},
    'FAIXA_ETARIA_3': {0: 'não 45-59', 1: '45-59'},
    'FAIXA_ETARIA_4': {0: 'não 60+', 1: '60+'},
    'ESPECTRO_POL_Direita': {0: 'não', 1: 'sim'},
    'ESPECTRO_POL_Esquerda': {0: 'não', 1: 'sim'},
    'ANO_ELEICAO': {2014: '2014', 2018: '2018', 2022: '2022'},
    'DS_ESTADO_CIVIL_CASADO(A)': {0: 'não', 1: 'sim'},
    'DS_ESTADO_CIVIL_OUTRO': {0: 'não', 1: 'sim'},
    'REGIAO_NORTE': {0: 'não', 1: 'sim'},
    'REGIAO_NORDESTE': {0: 'não', 1: 'sim'},
    'REGIAO_CENTRO-OESTE': {0: 'não', 1: 'sim'},
    'REGIAO_SUL': {0: 'não', 1: 'sim'},

}

# Processa somente colunas que existem no df_estad
colunas_categoricas = [c for c in mapas_categorias.keys() if c in df_estad.columns]

resumos = []
for col in colunas_categoricas:
    counts = df_estad[col].value_counts(dropna=False).sort_index()
    percents = (counts / counts.sum() * 100).round(2)

    # aplica mapeamento textual
    mapa = mapas_categorias[col]
    categorias = [mapa.get(idx, str(idx)) for idx in counts.index]

    resumo = pd.DataFrame({
        'variavel': col,
        'categoria': categorias,
        'count': counts.values,
        'percentual (%)': percents.values
    })
    resumos.append(resumo)

# Concatenar e salvar
df_resumo_final = pd.concat(resumos, ignore_index=True)
df_resumo_final.to_excel('resumo_categorias_estaduais.xlsx', index=False)


#%% Separando variáveis X e y
X = df_estad.drop(columns=['DS_SIT_TOT_TURNO', 'ANO_ELEICAO'])
y = df_estad['DS_SIT_TOT_TURNO']

print(y.value_counts())

#%% Separando amostras de treino e teste
X_train = X[df_estad['ANO_ELEICAO'].isin([2014, 2018])]
y_train = y[df_estad['ANO_ELEICAO'].isin([2014, 2018])]

X_test = X[df_estad['ANO_ELEICAO'] == 2022]
y_test = y[df_estad['ANO_ELEICAO'] == 2022]


#%% Estimando a Random Forest com balanceamento

rf_clf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=20,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=10,
    class_weight='balanced',  # balanceamento pelo scikit-learn
    random_state=100
)

rf_clf.fit(X_train, y_train)
## hiperparâmetros:

# n_estimators: qtde de árvores estimadas
# max_depth: profundidade máxima das árvores
# max_features: qtde de variáveis preditoras consideradas nos splits
# min_samples_split: qtde mínima de observações exigidas para dividir o nó
# min_samples_leaf: qtde mínima de observações exigidas para ser nó folha

#%% Obtendo os valores preditos pela RF

# Predict na base de treinamento
rf_pred_train_class = rf_clf.predict(X_train)
rf_pred_train_prob = rf_clf.predict_proba(X_train)

# Predict na base de testes
rf_pred_test_class = rf_clf.predict(X_test)
rf_pred_test_prob = rf_clf.predict_proba(X_test)

#%% Matriz de confusão (base de treino)

rf_cm_train = confusion_matrix(y_train, rf_pred_train_class)
cm_rf_train = ConfusionMatrixDisplay(rf_cm_train)

plt.rcParams['figure.dpi'] = 600
cm_rf_train.plot(colorbar=False, cmap='Oranges', values_format="d")
plt.title('Random Forest est.: Treino')
plt.xlabel('Classificado (Modelo)')
plt.ylabel('Observado (Real)')
plt.show()

acc_rf_train = accuracy_score(y_train, rf_pred_train_class)
sens_rf_train = recall_score(y_train, rf_pred_train_class, pos_label=1)
espec_rf_train = recall_score(y_train, rf_pred_train_class, pos_label=0)
prec_rf_train = precision_score(y_train, rf_pred_train_class)
f1_rf_train = f1_score(y_train, rf_pred_train_class)

print("Avaliação da RF (Base de Treino)")
print(f"Acurácia: {acc_rf_train:.1%}")
print(f"Sensibilidade: {sens_rf_train:.1%}")
print(f"Especificidade: {espec_rf_train:.1%}")
print(f"Precision: {prec_rf_train:.1%}")
print(f"F1-score: {f1_rf_train:.1%}")
print("\nRelatório completo:")
print(classification_report(y_train, rf_pred_train_class, digits=3))

#%% Matriz de confusão (base de teste)

rf_cm_test = confusion_matrix(y_test, rf_pred_test_class)
cm_rf_test = ConfusionMatrixDisplay(rf_cm_test)

plt.rcParams['figure.dpi'] = 600
cm_rf_test.plot(colorbar=False, cmap='Oranges', values_format="d")

plt.title('Random Forest est.: Teste')
plt.xlabel('Classificado (Modelo)')
plt.ylabel('Observado (Real)')
plt.show()

acc_rf_test = accuracy_score(y_test, rf_pred_test_class)
sens_rf_test = recall_score(y_test, rf_pred_test_class, pos_label=1)
espec_rf_test = recall_score(y_test, rf_pred_test_class, pos_label=0)
prec_rf_test = precision_score(y_test, rf_pred_test_class)
f1_rf_test = f1_score(y_test, rf_pred_test_class)

print("Avaliação da RF (Base de Teste)")
print(f"Acurácia: {acc_rf_test:.1%}")
print(f"Sensibilidade: {sens_rf_test:.1%}")
print(f"Especificidade: {espec_rf_test:.1%}")
print(f"Precision: {prec_rf_test:.1%}")
print(f"F1-score: {f1_rf_test:.1%}")
print("\nRelatório completo:")
print(classification_report(y_test, rf_pred_test_class, digits=3))


#%% ################# Grid Search CV ########################################

# best_params={'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 500}

# Lista de hiperparâmetros e seus valores para teste
param_grid_rf = {
    'n_estimators': [300, 400, 500, 600],
    'max_depth': [5, 10, 15, 20, 25],
    'max_features': ['sqrt', 'log2', 'none'],
    'min_samples_split': [2, 10, 50],
    'min_samples_leaf': [1, 5, 10]
}

# Identificar o algoritmo em uso
rf_grid = RandomForestClassifier(random_state=100, class_weight='balanced')

# Treinar os modelos para o grid search
rf_grid_model = GridSearchCV(estimator=rf_grid, 
                             param_grid=param_grid_rf,
                             scoring='f1',  # outras opções: 'accuracy', 'roc_auc', 'recall'
                             cv=5,
                             verbose=1,
                             n_jobs=-1)


# Treinamento 
rf_grid_model.fit(X_train, y_train)


#%% Verificando os melhores parâmetros obtidos

rf_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros
rf_best = rf_grid_model.best_estimator_

# Predict na base de treino
rf_grid_pred_train_class = rf_best.predict(X_train)
rf_grid_pred_train_prob = rf_best.predict_proba(X_train)

# Predict na base de testes
rf_grid_pred_test_class = rf_best.predict(X_test)
rf_grid_pred_test_prob = rf_best.predict_proba(X_test)

print(rf_grid_model.best_params_)


#%% Matriz de confusão (base de treino)

rf_grid_cm_train = confusion_matrix(y_train, rf_grid_pred_train_class)
cm_rf_grid_train = ConfusionMatrixDisplay(rf_grid_cm_train)

plt.rcParams['figure.dpi'] = 600
cm_rf_grid_train.plot(colorbar=False, cmap='Oranges', values_format="d")
plt.title('Random Forest est.: Treino (Após Grid Search)')
plt.xlabel('Classificado (Modelo)')
plt.ylabel('Observado (Real)')
plt.show()

acc_rf_grid_train = accuracy_score(y_train, rf_grid_pred_train_class)
sens_rf_grid_train = recall_score(y_train, rf_grid_pred_train_class, pos_label=1)
espec_rf_grid_train = recall_score(y_train, rf_grid_pred_train_class, pos_label=0)
prec_rf_grid_train = precision_score(y_train, rf_grid_pred_train_class)
f1_rf_grid_train = f1_score(y_train, rf_grid_pred_train_class)

print("Avaliação da RF Após Grid Search (Base de Treino)")
print(f"Acurácia: {acc_rf_grid_train:.1%}")
print(f"Sensibilidade: {sens_rf_grid_train:.1%}")
print(f"Especificidade: {espec_rf_grid_train:.1%}")
print(f"Precision: {prec_rf_grid_train:.1%}")
print(f"F1-score: {f1_rf_grid_train:.1%}")
print("\nRelatório completo:")
print(classification_report(y_train, rf_grid_pred_train_class, digits=3))

#%% Matriz de confusão (base de teste)

rf_grid_cm_test = confusion_matrix(y_test, rf_grid_pred_test_class)
cm_rf_grid_test = ConfusionMatrixDisplay(rf_grid_cm_test)

plt.rcParams['figure.dpi'] = 600
cm_rf_grid_test.plot(colorbar=False, cmap='Oranges', values_format="d")

plt.title('Random Forest est.: Teste (Após Grid Search)')
plt.xlabel('Classificado (Modelo)')
plt.ylabel('Observado (Real)')
plt.show()

acc_rf_grid_test = accuracy_score(y_test, rf_grid_pred_test_class)
sens_rf_grid_test = recall_score(y_test, rf_grid_pred_test_class, pos_label=1)
espec_rf_grid_test = recall_score(y_test, rf_grid_pred_test_class, pos_label=0)
prec_rf_grid_test = precision_score(y_test, rf_grid_pred_test_class)
f1_rf_grid_test = f1_score(y_test, rf_grid_pred_test_class)

print("Avaliação da RF Após Grid Search (Base de Teste)")
print(f"Acurácia: {acc_rf_grid_test:.1%}")
print(f"Sensibilidade: {sens_rf_grid_test:.1%}")
print(f"Especificidade: {espec_rf_grid_test:.1%}")
print(f"Precision: {prec_rf_grid_test:.1%}")
print(f"F1-score: {f1_rf_grid_test:.1%}")
print("\nRelatório completo:")
print(classification_report(y_test, rf_grid_pred_test_class, digits=3))

#%% Importância das variáveis preditoras

rf_features = pd.DataFrame({
    'features': X.columns.tolist(),
    'importance': rf_best.feature_importances_
})

rf_features = rf_features.sort_values(by='importance', ascending=False)

print(rf_features)
#%%
# Salvar tabela
rf_features.to_excel("RF_importancia_estad.xlsx", index=False)

#%% Curva ROC (base de teste)

# Parametrizando a função da curva ROC (real vs. previsto)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf_grid_pred_test_prob[:,1]) 
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plotando a curva ROC
plt.figure(figsize=(15,10), dpi=600)
plt.plot(fpr_rf, tpr_rf, color='Orange', linewidth=4)
plt.plot(fpr_rf, fpr_rf, color='gray', linestyle='dashed')
plt.title('AUC-ROC Random Forest est.: %g' % round(roc_auc_rf, 3), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensibilidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

#%% FIM da RF - base estadual

#%% Reg Log Bin - estadual (treino/teste), GLM com balanceamento 
#%% Importação dos pacotes
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

#%% Abrir a base
df_estad = pd.read_excel("df_estad.xlsx")
df_estad.info()

#%% Limpeza dos nomes
import unidecode
def limpar_nomes_colunas(colunas):
    colunas_novas = []
    for col in colunas:
        col = unidecode.unidecode(col)
        col = col.replace("–", "-").replace("-", "_")
        col = col.replace(" ", "_").replace("+", "_mais")
        col = col.replace(".", "").replace("(", "").replace(")", "")
        colunas_novas.append(col)
    return colunas_novas

df_estad.columns = limpar_nomes_colunas(df_estad.columns)

#%% Separação treino e teste (treino=2014+2018 | teste=2022)
df_train = df_estad[df_estad['ANO_ELEICAO'].isin([2014, 2018])].copy()
df_test  = df_estad[df_estad['ANO_ELEICAO'] == 2022].copy()

#%% Definição do conjunto de variáveis preditoras
cols_pred = [c for c in df_train.columns if c not in ['DS_SIT_TOT_TURNO', 'ANO_ELEICAO']]

#%% Ajuste do modelo GLM com pesos balanceados (na base de treino)

# Pesos balanceados equivalentes ao class_weight='balanced' (binário)
counts_train = df_train['DS_SIT_TOT_TURNO'].value_counts()
n_total_train = len(df_train)
n_classes = len(counts_train)
pesos_train = df_train['DS_SIT_TOT_TURNO'].apply(lambda x: n_total_train / (n_classes * counts_train[x]))

# X e y treino
X_train = sm.add_constant(df_train[cols_pred])
y_train = df_train['DS_SIT_TOT_TURNO']

#%% Teste de presença de multicolinearidade (VIF) na base de treino
from statsmodels.stats.outliers_influence import variance_inflation_factor

if X_train.shape[1] > 1:
    vif_df = pd.DataFrame({
        "variavel": X_train.columns,
        "VIF": [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    })
    vif_df = vif_df[vif_df['variavel'] != 'const'].sort_values('VIF', ascending=False)
    print("\nVIF (ordem decrescente) - Treino:")
    print(vif_df.to_string(index=False))

#%% Ajuste do modelo final (GLM Binomial com balanceamento)
modelo_final = sm.GLM(
    y_train,
    X_train,
    family=sm.families.Binomial(),
    freq_weights=pesos_train
).fit()

print(modelo_final.summary())


#%% Salvar os coeficientes (treino)
summary_df = modelo_final.summary2().tables[1].copy()

# Acrescenta OR e IC 95%
summary_df['OR'] = np.exp(summary_df['Coef.'])
if {'[0.025', '0.975]'}.issubset(summary_df.columns):
    summary_df['OR_2.5%'] = np.exp(summary_df['[0.025'])
    summary_df['OR_97.5%'] = np.exp(summary_df['0.975]'])
summary_df.to_excel("resultado_glm_est.xlsx", index=True)

#%% Tabela sintética (apenas variáveis significantes)

# summary_df 
tabela = summary_df.copy()
tabela = tabela.rename_axis('Variavel').reset_index()

# garante OR 
if 'OR' not in tabela.columns:
    tabela['OR'] = np.exp(tabela['Coef.'])

# filtra significantes e remove intercepto
tabela = tabela[(tabela['Variavel'] != 'const') & (tabela['P>|z|'] < 0.05)].copy()

# calcula probabilidade aproximada a partir do OR e texto de odds
tabela['Probabilidade (%)'] = 100 * (tabela['OR'] / (1 + tabela['OR']))
tabela['Chances (odds)'] = (np.round(tabela['OR'], 2)).astype(str) + ' : 1'

# renomeia e seleciona colunas finais
tabela = tabela.rename(columns={'Coef.': 'Coeficiente', 'P>|z|': 'P-valor'})
tabela = tabela[['Variavel', 'Coeficiente', 'P-valor', 'OR', 'Chances (odds)', 'Probabilidade (%)']]

# formatação básica
tabela['Coeficiente'] = tabela['Coeficiente'].round(3)
tabela['OR'] = tabela['OR'].round(3)
tabela['Probabilidade (%)'] = tabela['Probabilidade (%)'].round(1)
tabela['P-valor'] = tabela['P-valor'].map(lambda x: f"{x:.3g}")

# ordenar por significância
tabela = tabela.sort_values(by='P-valor')

# salva
tabela.to_excel("tabela_sintetica_glm_est.xlsx", index=False)
print("\nTabela sintética salva em: tabela_sintetica_glm_est.xlsx")


#%% Funções utilitárias (mantidas)
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score,
    precision_score, f1_score, ConfusionMatrixDisplay,
    roc_curve, auc
)

def matriz_confusao_completa(predicts, observado, cutoff=0.5):
    pred_bin = (predicts >= cutoff).astype(int)
    cm = confusion_matrix(observado, pred_bin)
    tn, fp, fn, tp = cm.ravel()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='viridis', colorbar=True, values_format='d')
    plt.xlabel('Classificado (Modelo)')
    plt.ylabel('Observado (real)')
    plt.title(f"Matriz de Confusão (cutoff={cutoff})")
    plt.show()

    return pd.DataFrame({
        'VP': [tp],
        'FP': [fp],
        'VN': [tn],
        'FN': [fn],
        'Acurácia': [accuracy_score(observado, pred_bin)],
        'Sensibilidade (Recall)': [recall_score(observado, pred_bin)],
        'Especificidade': [tn / (tn + fp) if (tn + fp) > 0 else np.nan],
        'Precision': [precision_score(observado, pred_bin, zero_division=0)],
        'F1-score': [f1_score(observado, pred_bin, zero_division=0)]
    })

def espec_sens(observado, predicts):
    values = np.asarray(predicts).ravel()
    y_true = np.asarray(observado).ravel()
    cutoffs = np.arange(0, 1.01, 0.01)

    lista_sensitividade, lista_especificidade = [], []
    for cutoff in cutoffs:
        pred_bin = (values >= cutoff).astype(int)
        sens = recall_score(y_true, pred_bin, pos_label=1)
        espec = recall_score(y_true, pred_bin, pos_label=0)
        lista_sensitividade.append(sens)
        lista_especificidade.append(espec)

    return pd.DataFrame({
        'cutoffs': cutoffs,
        'sensitividade': lista_sensitividade,
        'especificidade': lista_especificidade
    })



#%% Avaliação na base de TREINO (overfitting check)
df_train['phat_final_train'] = modelo_final.predict(X_train)

print("Avaliação na base de treino (2014+2018):")
print(matriz_confusao_completa(
    predicts=df_train['phat_final_train'],
    observado=y_train,
    cutoff=0.5
))
#%% Curva ROC e GINI no TREINO
fpr_tr, tpr_tr, _ = roc_curve(y_train, df_train['phat_final_train'])
roc_auc_tr = auc(fpr_tr, tpr_tr)
gini_tr = (roc_auc_tr - 0.5) / 0.5

plt.figure(figsize=(12, 8))
plt.plot(fpr_tr, tpr_tr, marker='o', linewidth=2)
plt.plot(fpr_tr, fpr_tr, linestyle='dashed', color='gray')
plt.title(f'ROC - Treino (2014+2018) | AUC: {roc_auc_tr:.4f} | GINI: {gini_tr:.4f}', fontsize=18)
plt.xlabel('1 - Especificidade', fontsize=14)
plt.ylabel('Sensitividade', fontsize=14)
plt.grid(True)
plt.show()

#%% Sensibilidade/Especificidade por cutoff na base de TREINO
dados_plotagem_tr = espec_sens(
    observado=y_train,
    predicts=df_train['phat_final_train']
)
plt.figure(figsize=(12,8))
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.plot(dados_plotagem_tr.cutoffs, dados_plotagem_tr.sensitividade, marker='o', label='Sensitividade (Treino)')
    plt.plot(dados_plotagem_tr.cutoffs, dados_plotagem_tr.especificidade, marker='o', label='Especificidade (Treino)')
plt.xlabel('Cutoff', fontsize=14)
plt.ylabel('Sensitividade / Especificidade', fontsize=14)
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=12)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
plt.legend(fontsize=14)
plt.title('Sensitividade/Especificidade por Cutoff - Treino (2014+2018)')
plt.show()

#%% Previsão na base de teste (2022)
X_test = sm.add_constant(df_test[cols_pred], has_constant='add')
df_test['phat_final'] = modelo_final.predict(X_test)

#%% Avaliação na base de teste
print("Avaliação na base de teste (ano 2022):")
print(matriz_confusao_completa(
    predicts=df_test['phat_final'],
    observado=df_test['DS_SIT_TOT_TURNO'],
    cutoff=0.5
))

#%% Curva ROC e GINI (teste)
fpr_test, tpr_test, _ = roc_curve(df_test['DS_SIT_TOT_TURNO'], df_test['phat_final'])
roc_auc_test = auc(fpr_test, tpr_test)
gini_test = (roc_auc_test - 0.5) / 0.5

plt.figure(figsize=(15, 10))
plt.plot(fpr_test, tpr_test, marker='o', linewidth=3)
plt.plot(fpr_test, fpr_test, linestyle='dashed', color='gray')
plt.title(f'ROC - Teste 2022 | AUC: {round(roc_auc_test, 4)} | GINI: {round(gini_test, 4)}', fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.grid(True)
plt.show()

#%% Sensitividade e Especificidade em função do cutoff (base de teste 2022)
dados_plotagem = espec_sens(
    observado=df_test['DS_SIT_TOT_TURNO'],
    predicts=df_test['phat_final']
)

plt.figure(figsize=(15,10))
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.plot(dados_plotagem.cutoffs, dados_plotagem.sensitividade,
             marker='o', color='indigo', markersize=8)
    plt.plot(dados_plotagem.cutoffs, dados_plotagem.especificidade,
             marker='o', color='limegreen', markersize=8)
plt.xlabel('Cutoff', fontsize=20)
plt.ylabel('Sensitividade / Especificidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.legend(['Sensitividade', 'Especificidade'], fontsize=20)
plt.title('Sensitividade/Especificidade por Cutoff - Teste 2022 (Estadual)')
plt.show()

#%% fim da reg log bin (treino 2014+2018 / teste 2022) com GLM balanceado 
#%% grafico de odds

#%% Importar pacotes
import pandas as pd
import matplotlib.pyplot as plt

#%% Ler base
df = pd.read_excel("tabela_sintetica_glm_est.xlsx")

#%% Ordenar por OR (opcional)
df = df.sort_values("OR")

#%% Plot com pontos e intervalos
plt.figure(figsize=(8, 6))

plt.errorbar(
    x=df["OR"],
    y=df["Variaveis"],
    xerr=[df["OR"] - df["OR_2.5%"], df["OR_97.5%"] - df["OR"]],
    fmt='o',
    color='black',
    ecolor='gray',
    elinewidth=1.2,
    capsize=3
)

# Linha de referência em OR=1
plt.axvline(x=1, color="red", linestyle="--")

plt.xlabel("Odds Ratio (OR)")
plt.ylabel("Variáveis")
plt.title("Odds Ratios com Intervalo de Confiança (95%)")

plt.tight_layout()
plt.show()
#%%