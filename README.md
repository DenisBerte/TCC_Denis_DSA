
# Candidaturas religiosas e de segurança nas eleições brasileiras para deputados (2014–2022)

Este repositório contém os scripts, dados brutos, processados e materiais do meu Trabalho de Conclusão de Curso (TCC) em Data Science e Analytics (ESALQ).
---

## Resumo do projeto

Título: Candidaturas religiosas e de segurança nas eleições brasileiras para deputados (2014-2022): investigando a probabilidade de ser eleito(a)

Muitos candidatos a cargos políticos utilizam como nome de urna termos que servem como dicas, com as quais esperam influenciar o comportamento de voto do eleitorado. A partir da última década, cresce a atenção voltada a candidaturas políticas ligadas ao campo religioso e da segurança pública no Brasil.  

A questão principal foi: **ser candidato com identidade religiosa ou da segurança tem efeito sobre a probabilidade de eleição de deputados federais ou estaduais?**  

Além disso, o estudo comparou duas técnicas de análise:  
- **Random Forest (RF)**, um método de aprendizado de máquina não paramétrico.  
- **Regressão Logística Binária (Logit)**, uma técnica estatística tradicional.  

Para ambos os modelos foram utilizados dados das eleições de **2014 e 2018** (treino) e **2022** (teste), mantendo separados os candidatos a deputados estaduais e federais.  

Os resultados mostraram **efeitos fracos ou moderados** das variáveis “religioso” e “segurança” sobre o resultado eleitoral nos anos analisados. Ainda assim, o desempenho dos modelos foi bastante semelhante, com **vantagem marginal do Random Forest** sobre a regressão logística binária.  

**Palavras-chave:** eleição de deputados; nome de urna; ocupação; candidato religioso; candidato segurança.

---

## Estrutura do repositório

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
