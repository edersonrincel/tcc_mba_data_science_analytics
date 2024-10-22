#Pacotes utilizados
pacotes <- c("plotly","tidyverse","knitr","kableExtra","fastDummies","rgl","car",
             "reshape2","jtools","stargazer","lmtest","caret","pROC","ROCR","nnet",
             "magick","cowplot","globals","equatiomatic")

options(rgl.debug = TRUE)

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T) 
} else {
  sapply(pacotes, require, character = T) 
}

#Carrega dados simulados
dados <- read.csv("1.base_tratada.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

#Normaliação da base
# Função para normalizar variáveis categóricas com one-hot encoding
dados_tratados <- dados %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.factor), ~ as.numeric(as.factor(.))))

# Normalização das variáveis que agora são numéricas
numeric_columns <- sapply(dados_tratados, is.numeric)

# Excluir a variável target 'possui_cartao_black' da normalização
numeric_columns["possui_cartao_black"] <- FALSE

#Aplicação da função scale para normalização
dados_normalizados <- dados_tratados
dados_normalizados[numeric_columns] <- scale(dados_tratados[numeric_columns])

# Verificar amostra da base normalizada
head(dados_normalizados)

#salvar a base normalizada em um arquivo .csv para utilização no modelo
write.csv(dados_normalizados, "2.base_normalizada.csv", row.names = FALSE)

