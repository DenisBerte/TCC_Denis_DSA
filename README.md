# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 20:05:20 2025

@author: denis
"""

# Candidaturas religiosas e de segurança nas eleições brasileiras para deputados (2014–2022)

Este repositório contém os scripts, dados processados e materiais do meu Trabalho de Conclusão de Curso (TCC), que investiga se candidatos a deputados estaduais e federais que utilizaram **nomes de urna** ou tinham **ocupação ligada à religião ou à segurança pública** apresentaram maior probabilidade de eleição.

---

## Resumo do projeto

Muitos candidatos a cargos políticos utilizam como nome de urna termos que servem como dicas, com as quais esperam influenciar o comportamento de voto do eleitorado. A partir da última década, cresce a atenção voltada a candidaturas políticas ligadas ao campo religioso e da segurança pública no Brasil.  

O objetivo geral deste estudo foi verificar se os candidatos a deputados estaduais e federais que utilizaram nomes de urna ou tinham ocupação nessas áreas no ano da eleição tiveram maior probabilidade de eleição. A questão principal foi: **utilizar nome de urna e/ou ter ocupação religiosa ou de segurança aumenta a chance de o candidato ser eleito deputado?**  

Além disso, o estudo comparou duas técnicas de análise:  
- **Random Forest (RF)**, um método de aprendizado de máquina não paramétrico.  
- **Regressão Logística Binária (Logit)**, uma técnica estatística tradicional.  

Para ambos os modelos foram utilizados dados das eleições de **2014 e 2018** (treino) e **2022** (teste), mantendo separados os candidatos a deputados estaduais e federais.  

Os resultados mostraram a **baixa capacidade explicativa isolada** das variáveis “religioso” e “segurança” sobre o resultado eleitoral. Ainda assim, o desempenho dos modelos foi bastante semelhante, com **vantagem marginal do Random Forest** sobre a regressão logística binária.  

**Palavras-chave:** eleição de deputados; nome de urna; ocupação; candidato religioso; candidato segurança.

---

## 📂 Estrutura do repositório

meu-tcc/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ scripts/
│ ├─ 00_funcoes_ajuda.py # funções auxiliares
│ ├─ 01_manipulacao_geral.py # gera bases processadas
│ ├─ 02_analises_dep_estadual.py # análises deputados estaduais
│ └─ 03_analises_dep_federal.py # análises deputados federais
├─ data/
│ ├─ raw/ # dados originais (3 arquivos .xlsx)
│ └─ processed/ # dados processados (5 arquivos .xlsx gerados)
├─ outputs/
│ ├─ figs/ # gráficos gerados
│ └─ tables/ # tabelas exportadas


---
