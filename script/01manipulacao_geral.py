# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 14:33:04 2025

@author: denis
"""

#%%
# Importando as bibliotecas
from pyspark import SparkConf
from pyspark.sql import SparkSession
import findspark
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import numpy as np
from pyspark.sql import functions as F
import pyarrow as pya

#%% #####AVISO######
# Inicialmente, este projeto incluia dados municipais com arquivos muito extensos,
#por isso foi utilizado o pyspark. porém, como agora o código
#manipula apenas 3 bases de eleições para deputados, o spark não é necessário.
#portanto, o código que segue pode ser atualizado para rodar fora do ambiente spark,
#ou podem ser incluídos dados municipais que são arquivos muito grandes


#%% Encontrando as configurações para inicializar o Spark

findspark.init()

#%% Spark

# Nome da aplicação Spark
app_name = "Aplicação Spark"

# Inicializando o objeto de configurações do Spark
conf = SparkConf()

# Configuração do tipo de cluster
conf.set("spark.master", "local[*]")
# Definição do nome da aplicação
conf.set("spark.app.name", app_name)
# Definição da quantidade de núcleos de processamento a serem usados
conf.set("spark.executor.cores", "2")
# Definição da quantidade de memória alocada para cada executor
conf.set("spark.executor.memory", "2g")
conf.set("spark.driver.memory", "1g")

# Criando a sessão Spark
spark = SparkSession.builder.config(conf=conf).getOrCreate()

print("Spark inicializado com sucesso!")

#%% Bases
# No site do TSE esses são os nomes originais das bases:
#consulta_cand_2014_BRASIL
#consulta_cand_2018_BRASIL
#consulta_cand_2022_BRASIL

# Lendo os dataFrames Spark a partir de arquivo CSV

#dep
df14 = spark.read.csv("base_2014.csv", header=True, sep=";", inferSchema=True, encoding="latin1")
df18 = spark.read.csv("base_2018.csv", header=True, sep=";", inferSchema=True, encoding="latin1")
df22 = spark.read.csv("base_2022.csv", header=True, sep=";", inferSchema=True, encoding="latin1")

#%% detalhes das bases

df14.printSchema()
df18.printSchema()
df22.printSchema()

#%%

# Lista de variaveis das bases originais para manter
var_selec = [
    "ANO_ELEICAO",
    "SG_UF",
    "DS_CARGO",
    "DT_NASCIMENTO", 
    "NR_CANDIDATO",
    "NM_CANDIDATO",
    "NM_URNA_CANDIDATO",
    "NR_PARTIDO",
    "SG_PARTIDO",
    "NM_PARTIDO",
    "DS_GENERO",
    "DS_GRAU_INSTRUCAO",
    "DS_ESTADO_CIVIL",
    "DS_COR_RACA",
    "DS_OCUPACAO",
    "DS_SIT_TOT_TURNO",
]

#%%
# Selecionando apenas as var selecionadas

df14 = df14.select(var_selec)
df18 = df18.select(var_selec)
df22 = df22.select(var_selec)



#%%
# Criando dfdep com os anos 2014, 2018, 2022

dfdep = df14.unionByName(df18, allowMissingColumns=True) \
            .unionByName(df22, allowMissingColumns=True)
            
# contando        
dfdep.groupBy("ANO_ELEICAO").count().show()

#%% detalhes da base

dfdep.printSchema()


#%% detalhes descritivos das var e categorias

# Lista de colunas fornecida
colunas = [
    "ANO_ELEICAO",
    "SG_UF",
    "DS_CARGO",
    "DT_NASCIMENTO", 
    "NR_CANDIDATO",
    "NM_CANDIDATO",
    "NM_URNA_CANDIDATO",
    "NR_PARTIDO",
    "SG_PARTIDO",
    "NM_PARTIDO",
    "DS_GENERO",
    "DS_GRAU_INSTRUCAO",
    "DS_ESTADO_CIVIL",
    "DS_COR_RACA",
    "DS_OCUPACAO",
    "DS_SIT_TOT_TURNO",
]

# Para cada coluna, mostrar os valores únicos e suas contagens
for col in colunas:
    print(f"\nValores distintos para a coluna: {col}")
    dfdep.groupBy(col).count().orderBy("count", ascending=False).show(truncate=False)

#%% Filtrar base para manter apenas deputados estaduais e federais 
# e sobrescrever a base

dfdep = dfdep.filter(dfdep["DS_CARGO"].isin(["DEPUTADO ESTADUAL", "DEPUTADO FEDERAL"]))

#conferir
dfdep.select("DS_CARGO").distinct().show()

#contar
dfdep.groupBy("DS_CARGO").count().orderBy("count", ascending=False).show(truncate=False)

#contar por ano
dfdep.groupBy("ANO_ELEICAO", "DS_CARGO").count().orderBy("ANO_ELEICAO", "DS_CARGO").show(truncate=False)


#%% Definir padrões de palavras-chave para classificar os candidatos religiosos

religiosos = [
    r"^APÓSTOLO", r"^APOSTOLO", r"^APÓSTOLA", r"^APOSTOLA", r"^BISPO", r"^BISPA", 
    r"^CAPELÃO", r"^CAPELAO", r"^DOM\s", r"^IRMÃ", r"^IRMÃO", r"^IRMAO", 
    r"MISSIONÁRIO", r"MISSIONARIO", r"MISSIONÁRIA", r"MISSIONARIA", 
    r"o PASTOR", r"P\.B\.", r"PADRE", r"PASTOR\s", r"PASTORA\s", r"PASTOR$", 
    r"PR\.", r"^PR\s", r"^PRA\s", r"^PRA\.", r"^REV\s", r"^REV\.", 
    r"^REVERENDO", r"PRESBÍTERO", r"^PRESB\.", r"^PAI", r"^MÃE", 
    r"^MINISTRO", r"^MINISTRA", r"^ABENÇOADO", r"^ABENÇOADA", r"^BABALORIXÁ", 
    r"^IALORIXÁ", r"^OGUM", r"^EXÚ", r"^IANSÃ", r"^YANSÃ", r"^IEMANJÁ", 
    r"^OBALUAÊ", r"^OXALÁ", r"^OMULU", r"^OXÓSSI", r"^OXUM", r"^OXUMARÉ", 
    r"^XANGÔ"
]

#%% classificar candidatos segurança

seguranca = [
    r"^AGENTE", r"^AGT\.", r"ASPIRANTE", r"^ASPIRANTE A OFICIAL", r"BOMBEIRO", r"BOMBEIRA", 
    r"^CABO", r"CAPITÃO", r"CAPITAO", r"^CAP\.", r"^CAP ", r"^CAPITÂ", r"^CAPITA", 
    r"^CB", r"^CB\.", r"^CEL\s", r"^CEL\.", r"^CORONEL", r"COMANDANTE", r"^CTE ", 
    r"DELE\.", r"DELEGADA", r"DELEGADO", r"DETETIVE", r"GCM", r"GCM\.", r"^GENERAL", 
    r"INSPETOR", r"INSTRUTOR", r"INVESTIGADOR", r"MAJOR", r"PM", r"POLICIAL", 
    r"POLICIALFEDERAL", r"SARGENTO", r"^SARGENTA", r"SARG\.", r"^SGTO", r"^SGT\.", 
    r"^SGT ", r"^SD", r"^SD\.", r"SOLDADO\s", r"^SUB", r"SUBO FICIAL", r"SUBOFICIAL", 
    r"^SUBTEN", r"^SUB TEN", r"^ST\s", r"SUBTENENTE", r"SUB TENENTE", r"^T C", 
    r"^T\. C\.", r"^T\. C\. CORONEL", r"^TEN\.", r"TENENTE", r"TENENTE CORONEL", 
    r"TC CORONEL", r"TCORONEL", r"TEN CORONEL", r"TEN\. CEL", r"TEN\. CORONEL",
    r"^GENERAL DE EXÉRCITO", r"^GENERAL DE DIVISÃO", r"^GENERAL DE BRIGADA", 
    r"^TENENTE-CORONEL", r"^PRIMEIRO-TENENTE", r"^SEGUNDO-TENENTE", r"^CADETE", 
    r"^PRIMEIRO-SARGENTO", r"^SEGUNDO-SARGENTO", r"^TERCEIRO-SARGENTO", 
    r"^ALMIRANTE", r"^ALMIRANTE DE ESQUADRA", r"^VICE-ALMIRANTE", r"^CONTRA-ALMIRANTE", 
    r"^CAPITÃO DE MAR E GUERRA", r"^CAPITÃO DE FRAGATA", r"^CAPITÃO DE CORVETA", 
    r"^CAPITÃO-TENENTE", r"^GUARDA-MARINHA", r"^MARINHEIRO", r"^MARECHAL DO AR", 
    r"^TENENTE-BRIGADEIRO", r"^MAJOR-BRIGADEIRO", r"^BRIGADEIRO", r"^PRAÇA", 
    r"^POLÍCIA"
]

#%% ocupação declarada na area da segurança

ocupacao_seg = [
    "POLICIAL CIVIL", "POLICIAL MILITAR", "BOMBEIRO CIVIL", "MILITAR REFORMADO", 
    "BOMBEIRO MILITAR", "MEMBRO DAS FORÇAS ARMADAS", "DELEGADO DE POLICIA",  
    "OFICIAIS DAS FORÇAS ARMADAS E FORÇAS AUXILIARES", "MILITAR EM GERAL"
]

#%% ocupação religiosa declarada 

ocupacao_reg = [
    "SACERDOTE OU MEMBRO DE ORDEM OU SEITA RELIGIOSA", 
    "SACERDOTE OU MEMBRO DE ORDENS OU SEITAS RELIGIOSAS"
]

#%% BASE DEP

# Criando colunas categorizando candidatos religiosos com base no nome (usando regex)
dfdep = dfdep.withColumn(
    "RELIGIOSO", 
    F.when(F.col("NM_URNA_CANDIDATO").rlike("|".join(religiosos)), 1).otherwise(0)
)

# Criando colunas categorizando candidatos da segurança com base no nome (usando regex)
dfdep = dfdep.withColumn(
    "SEGURANCA", 
    F.when(F.col("NM_URNA_CANDIDATO").rlike("|".join(seguranca)), 1).otherwise(0)
)

# Criando colunas categorizando candidatos com base na ocupação de segurança (usando regex)
dfdep = dfdep.withColumn(
    "OCUPACAO_SEG",
    F.when(F.col("DS_OCUPACAO").rlike("|".join(ocupacao_seg)), 1).otherwise(0)
)

# Criando colunas categorizando candidatos religiosos com base na ocupação (usando regex)
dfdep = dfdep.withColumn(
    "OCUPACAO_REG",
    F.when(F.col("DS_OCUPACAO").rlike("|".join(ocupacao_reg)), 1).otherwise(0)
)

#%% criando as duas variaveis binarias

# RELIGIOSOS = 1 se RELIGIOSO (nome) OU OCUPACAO_REG = 1
dfdep = dfdep.withColumn(
    "RELIGIOSOS",
    F.when((F.col("RELIGIOSO") == 1) | (F.col("OCUPACAO_REG") == 1), 1).otherwise(0)
)

# SEGURANCA = 1 se SEGURANCA (nome) OU OCUPACAO_SEG = 1
# (cria coluna temporária e renomeia para manter o nome SEGURANCA)
dfdep = (
    dfdep
    .withColumn(
        "SEGURANCA_CONSOL",
        F.when((F.col("SEGURANCA") == 1) | (F.col("OCUPACAO_SEG") == 1), 1).otherwise(0)
    )
    .drop("SEGURANCA")
    .withColumnRenamed("SEGURANCA_CONSOL", "SEGURANCA")
)


# Remover as colunas intermediárias antigas, mantendo apenas as finais
dfdep = dfdep.drop("RELIGIOSO", "OCUPACAO_SEG", "OCUPACAO_REG")

#%% detalhes da codificação 

dfdep.select("RELIGIOSOS").distinct().show(truncate=False)
dfdep.select("SEGURANCA").distinct().show(truncate=False)


#%% detalhes da(s) var(s) - resultado eleitoral e categorias

# Lista de colunas fornecida
variaveis = ["DS_SIT_TOT_TURNO"]

# Para cada coluna, mostrar os valores únicos e suas contagens
for col in variaveis:
    print(f"\nValores distintos para a coluna: {col}")
    dfdep.groupBy(col).count().orderBy("count", ascending=False).show(truncate=False)


#%% recodificar "DS_SIT_TOT_TURNO" (eleito x nao eleito)

from pyspark.sql import functions as F

# Categorias que devem virar "ELEITO"
categorias_eleito = ["ELEITO POR QP", "ELEITO POR MÉDIA"]

# Categorias que representam nulo
categorias_nulo = ["#NULO#", "#NULO"]

def recategorizar_situacao(df):
    return df.withColumn(
        "DS_SIT_TOT_TURNO",
        F.when(F.trim(F.col("DS_SIT_TOT_TURNO")).isin(categorias_eleito), "ELEITO")
         .when(F.trim(F.col("DS_SIT_TOT_TURNO")).isin(categorias_nulo), F.lit(None))
         .otherwise("NÃO ELEITO")
    )

# Aplicar a função ao DataFrame
dfdep = recategorizar_situacao(dfdep)



#%%

#remover null
dfdep = dfdep.filter(F.col("DS_SIT_TOT_TURNO").isNotNull())

#conferir
dfdep.groupBy("DS_SIT_TOT_TURNO").count().show(truncate=False)


#%%recod genero

#detalhes da var genero
dfdep.groupBy("DS_GENERO").count().orderBy("count", ascending=False).show(truncate=False)


# Lista de valores que queremos manter
generos_validos = ["MASCULINO", "FEMININO"]

# Filtrar dfdep para manter apenas os valores desejados
dfdep = dfdep.filter(F.col("DS_GENERO").isin(generos_validos))

#conferir
print("Distribuição de DS_GENERO em dfdep após filtro:")
dfdep.groupBy("DS_GENERO").count().show()



#%%recod escolaridade

#detalhes da var genero
dfdep.groupBy("DS_GRAU_INSTRUCAO").count().orderBy("count", ascending=False).show(truncate=False)

# Dicionário de recodificação dos graus de instrução
grau_instrucao_map = {
    "LÊ E ESCREVE": "Até EF",
    "ENSINO FUNDAMENTAL COMPLETO": "Até EF",
    "ENSINO FUNDAMENTAL INCOMPLETO": "Até EF",
    "ENSINO MÉDIO COMPLETO": "Até EM",
    "ENSINO MÉDIO INCOMPLETO": "Até EM",
    "SUPERIOR INCOMPLETO": "Até Sup.",
    "SUPERIOR COMPLETO": "Até Sup."
}

# Função para recategorizar o grau de instrução
def recategorizar_instrucao(df):
    return df.withColumn(
        "DS_GRAU_INSTRUCAO",
        F.when(F.col("DS_GRAU_INSTRUCAO").isin(["LÊ E ESCREVE", 
                                                "ENSINO FUNDAMENTAL COMPLETO", 
                                                "ENSINO FUNDAMENTAL INCOMPLETO"]), "Até EF")
        .when(F.col("DS_GRAU_INSTRUCAO").isin(["ENSINO MÉDIO COMPLETO", "ENSINO MÉDIO INCOMPLETO",
                                               ]), "Até EM")
        .when(F.col("DS_GRAU_INSTRUCAO").isin(["SUPERIOR INCOMPLETO", "SUPERIOR COMPLETO"]), "Até Sup.")
        .otherwise(F.col("DS_GRAU_INSTRUCAO"))  # Caso algum valor inesperado apareça, mantém como está
    )

# Aplicar a recategorização na base
dfdep = recategorizar_instrucao(dfdep)

# Verificar a distribuição após a recodificação
print("Distribuição de DS_GRAU_INSTRUCAO em dfdep:")
dfdep.groupBy("DS_GRAU_INSTRUCAO").count().show()


#%%recod estado civil

dfdep.groupBy("DS_ESTADO_CIVIL").count().orderBy("count", ascending=False).show(truncate=False)

# Função para recategorizar o estado civil
def recategorizar_estado_civil(df):
    return df.withColumn(
        "DS_ESTADO_CIVIL",
        F.when(F.col("DS_ESTADO_CIVIL").isin(["SOLTEIRO(A)", "CASADO(A)"]), F.col("DS_ESTADO_CIVIL"))
        .otherwise("OUTRO")  # Qualquer outro valor será transformado em "OUTRO"
    )

# Aplicar a recategorização
dfdep = recategorizar_estado_civil(dfdep)


print("Distribuição de DS_ESTADO_CIVIL em dfdep:")
dfdep.groupBy("DS_ESTADO_CIVIL").count().show()


#%% recod cor/raça 

dfdep.groupBy("DS_COR_RACA").count().orderBy("count", ascending=False).show(truncate=False)

# Lista de valores que devem ser removidos
valores_nulos = ["#NE#", "NÃO INFORMADO"]

# Função para recategorizar a variável DS_COR_RACA
def recategorizar_cor_raca(df):
    df = df.filter(~F.col("DS_COR_RACA").isin(valores_nulos))  # Remove os valores nulos

    df = df.withColumn(
        "DS_COR_RACA",
        F.when(F.col("DS_COR_RACA").isin("BRANCA", "AMARELA"), "BRANCA")
         .when(F.col("DS_COR_RACA").isin("PRETA", "PARDA", "INDÍGENA"), "NÃO BRANCA")
         .otherwise(None)  # segurança extra, caso apareça algo inesperado
    )
    return df

# Aplicar a recategorização 
dfdep = recategorizar_cor_raca(dfdep)

#conferir
print("Distribuição de DS_COR_RACA em dfdep:")
dfdep.groupBy("DS_COR_RACA").count().show()


#%% recod partido base dep - criação da var espectro-pol (Bolognesi et al., 2023).

# DEP
dfdep.groupBy("SG_PARTIDO").count().orderBy("count", ascending=False).show(truncate=False)

from pyspark.sql.functions import when, col

# Dicionário de classificação dos partidos (ajustado conforme a imagem)
classificacao_partidos = {
    # Esquerda
    "PSOL": "Esquerda", "PCO": "Esquerda", "PCB": "Esquerda", "PSTU": "Esquerda",
    "PT": "Esquerda", "PC do B": "Esquerda", "PDT": "Esquerda", "PSB": "Esquerda", "UP": "Esquerda",

    # Centro
    "REDE": "Centro", "PPS": "Centro", "PV": "Centro",

    # Direita
    "PTB": "Direita", "AVANTE": "Direita", "SD": "Direita", "PMN": "Direita",
    "PMB": "Direita", "PSD": "Direita", "PSDB": "Direita", "PMDB": "Direita",
    "PHS": "Direita", "PODE": "Direita", "PPL": "Direita", "PRTB": "Direita",
    "PTC": "Direita", "PR": "Direita", "PRP": "Direita", "PROS": "Direita",
    "DC": "Direita", "Progressistas": "Direita", "PSL": "Direita", "NOVO": "Direita",
    "DEM": "Direita", "PATRIOTA": "Direita", "PSC": "Direita", "PRB": "Direita",
    "PL": "Direita", "SOLIDARIEDADE": "Direita", "MDB": "Direita", "REPUBLICANOS": "Direita",
    "CIDADANIA": "Direita", "UNIÃO": "Direita", "AGIR": "Direita", "PT do B": "Direita",
    "PTN": "Direita", "PSDC": "Direita", "PP": "Direita"
}

# Criar a condição para recodificação (apenas para manter padrão, mesmo sem uso direto)
condicao = when(col("SG_PARTIDO").isin(list(classificacao_partidos.keys())), col("SG_PARTIDO"))

# Aplicar a classificação correta
dfdep = dfdep.withColumn(
    "ESPECTRO_POL",
    when(col("SG_PARTIDO") == "PSOL", "Esquerda")
    .when(col("SG_PARTIDO") == "PCO", "Esquerda")
    .when(col("SG_PARTIDO") == "PCB", "Esquerda")
    .when(col("SG_PARTIDO") == "PSTU", "Esquerda")
    .when(col("SG_PARTIDO") == "PT", "Esquerda")
    .when(col("SG_PARTIDO") == "PC do B", "Esquerda")
    .when(col("SG_PARTIDO") == "PDT", "Esquerda")
    .when(col("SG_PARTIDO") == "PSB", "Esquerda")
    .when(col("SG_PARTIDO") == "UP", "Esquerda")

    .when(col("SG_PARTIDO") == "REDE", "Centro")
    .when(col("SG_PARTIDO") == "PPS", "Centro")
    .when(col("SG_PARTIDO") == "PV", "Centro")

    .when(col("SG_PARTIDO") == "PTB", "Direita")
    .when(col("SG_PARTIDO") == "AVANTE", "Direita")
    .when(col("SG_PARTIDO") == "SD", "Direita")
    .when(col("SG_PARTIDO") == "PMN", "Direita")
    .when(col("SG_PARTIDO") == "PMB", "Direita")
    .when(col("SG_PARTIDO") == "PSD", "Direita")
    .when(col("SG_PARTIDO") == "PSDB", "Direita")
    .when(col("SG_PARTIDO") == "PMDB", "Direita")
    .when(col("SG_PARTIDO") == "PHS", "Direita")
    .when(col("SG_PARTIDO") == "PODE", "Direita")
    .when(col("SG_PARTIDO") == "PPL", "Direita")
    .when(col("SG_PARTIDO") == "PRTB", "Direita")
    .when(col("SG_PARTIDO") == "PTC", "Direita")
    .when(col("SG_PARTIDO") == "PR", "Direita")
    .when(col("SG_PARTIDO") == "PRP", "Direita")
    .when(col("SG_PARTIDO") == "PROS", "Direita")
    .when(col("SG_PARTIDO") == "DC", "Direita")
    .when(col("SG_PARTIDO") == "Progressistas", "Direita")
    .when(col("SG_PARTIDO") == "PSL", "Direita")
    .when(col("SG_PARTIDO") == "NOVO", "Direita")
    .when(col("SG_PARTIDO") == "DEM", "Direita")
    .when(col("SG_PARTIDO") == "PATRIOTA", "Direita")
    .when(col("SG_PARTIDO") == "PSC", "Direita")
    .when(col("SG_PARTIDO") == "PRB", "Direita")
    .when(col("SG_PARTIDO") == "PL", "Direita")
    .when(col("SG_PARTIDO") == "SOLIDARIEDADE", "Direita")
    .when(col("SG_PARTIDO") == "MDB", "Direita")
    .when(col("SG_PARTIDO") == "REPUBLICANOS", "Direita")
    .when(col("SG_PARTIDO") == "CIDADANIA", "Direita")
    .when(col("SG_PARTIDO") == "UNIÃO", "Direita")
    .when(col("SG_PARTIDO") == "AGIR", "Direita")
    .when(col("SG_PARTIDO") == "PT do B", "Direita")
    .when(col("SG_PARTIDO") == "PTN", "Direita")
    .when(col("SG_PARTIDO") == "PSDC", "Direita")
    .when(col("SG_PARTIDO") == "PP", "Direita")
)

# Verificar a distribuição após a recodificação
print("Distribuição do ESPECTRO POLITICO em dfdep:")
dfdep.groupBy("ESPECTRO_POL").count().show()



#%% recod DT_NASCIMENTO (para criar var idade e faixa etaria)

dfdep.select("DT_NASCIMENTO").show(10, truncate=False)

from pyspark.sql.functions import when, col, to_date, year

dfdep = dfdep.withColumn("DT_NASCIMENTO_DATE", to_date(col("DT_NASCIMENTO"), "dd/MM/yyyy"))
dfdep = dfdep.withColumn("ANO_NASCIMENTO", year(col("DT_NASCIMENTO_DATE")))
dfdep = dfdep.withColumn("IDADE", col("ANO_ELEICAO") - col("ANO_NASCIMENTO"))
dfdep = dfdep.withColumn("FAIXA_ETARIA",
    when(col("IDADE") < 30, "18–29")
    .when((col("IDADE") >= 30) & (col("IDADE") <= 44), "30–44")
    .when((col("IDADE") >= 45) & (col("IDADE") <= 59), "45–59")
    .when(col("IDADE") >= 60, "60+")
    .otherwise("Desconhecida")
)

#verificar
dfdep.groupBy("FAIXA_ETARIA").count().orderBy("FAIXA_ETARIA").show()


#%% recodificar estados em grande região - criar var grande região

dfdep.groupBy("SG_UF").count().orderBy("count", ascending=False).show(30, truncate=False)


from pyspark.sql.functions import when, col

def adicionar_regiao(df):
    return df.withColumn(
        "REGIAO",
        when(col("SG_UF").isin("AC", "AP", "AM", "PA", "RO", "RR", "TO"), "NORTE")
        .when(col("SG_UF").isin("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"), "NORDESTE")
        .when(col("SG_UF").isin("DF", "GO", "MS", "MT"), "CENTRO-OESTE")  # <- DF corretamente aqui
        .when(col("SG_UF").isin("ES", "MG", "RJ", "SP"), "SUDESTE")
        .when(col("SG_UF").isin("PR", "RS", "SC"), "SUL")
        .otherwise("DESCONHECIDA")
    )

#aplicar
dfdep = adicionar_regiao(dfdep)

#verificar
dfdep.groupBy("REGIAO").count().orderBy("count", ascending=False).show()

#%% salvar base como um df pandas para trabalhar fora do spark
dfdep = dfdep.toPandas()

dfdep.to_excel("dfdep.xlsx", index=False)

#%% Retirando o dataframe spark da memória e do disco

dfdep.unpersist()

# Limpar o cache
spark.catalog.clearCache()

# Finalizando a aplicação Spark
spark.stop()

#%%

#FIM DAS MANIPULAÇÕES das bases originais em ambiente SPARK

#%%