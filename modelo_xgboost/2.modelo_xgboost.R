#Pacotes utilizados
pacotes <- c("xgboost","MLmetrics","PRROC")

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

# Carrega dados gerados pelo script "2.tratamento.R"
data <- read.csv("2.base_normalizada.csv")

# Separar variáveis da variável target
X <- as.matrix(data[, -ncol(data)])  # Todas as colunas exceto a última
y <- data$possui_cartao_black        # Variável target

# Separar dados entre treino e teste 80% treino, 20% teste)
set.seed(123) #definir uma "semente" (seed) para o gerador de números aleatórios
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Treino do modelo XGBoost 
xgb_model <- xgboost(
  data = X_train,
  label = y_train,
  nrounds = 28, # O teste de cross-validation (abaixo) indica que o número ideal de rodadas é 28.
  objective = "binary:logistic",
  eval_metric = "error",
  verbose = 0
)

# Salvar o modelo em um arquivo .RData
save(xgb_model, file = "modelo_xgboost.RData")

# Fazer predições na base de testes
predictions <- predict(xgb_model, X_test)
predicted_classes <- ifelse(predictions > 0.35, 1, 0) #Ajustado de 0.5 para 0.35 para melhorar a Especificidade (detecção da classe minoritária)

# Aferição da acurácia
accuracy <- mean(predicted_classes == y_test)
print(paste("Acurácia:", accuracy)) 
#acurácia de 95%


# AVALIAÇÕES DO MODELO

#--------------------- Matriz de confusão ---------------------
confusion_matrix <- confusionMatrix(as.factor(predicted_classes), as.factor(y_test))
print(confusion_matrix)

#           Reference
# Prediction    0    1
#          0 2630   48
#          1   99  325
#
#                  Kappa : 0.7885 -> Um valor de Kappa acima de 0.7 é considerado "bom", o que indica que o modelo tem uma boa capacidade preditiva além do mero acaso.        
#            Sensitivity : 0.9637 -> O modelo acerta 96.37% dos casos onde a classe verdadeira é "0": não tem cartão black   
#            Specificity : 0.8713 -> O modelo acerta 87.13% dos casos onde a classe verdadeira é "1": tem cartão black      
#         Pos Pred Value : 0.9821 -> O modelo está prevendo corretamente a classe "1": tem cartão black, a maior parte do tempo.       
#         Neg Pred Value : 0.7665 -> O modelo tem uma razoável capacidade de prever corretamente a classe "0": não tem cartão black 
#      Balanced Accuracy : 0.9175 -> Indica que o modelo está equilibrado em prever ambas as classes



#--------------------- AUC-ROC (Área Sob a Curva ROC) ---------------------
# Carregar a biblioteca pROC para calcular e plotar a curva ROC
library(pROC)
# Gerar as probabilidades preditas no conjunto de teste
predictions_prob <- predict(xgb_model, X_test)
# Calcular a curva ROC e a AUC
roc_obj <- roc(y_test, predictions_prob)
# Exibir o valor da AUC
auc_value <- auc(roc_obj)
print(paste("AUC:", auc_value))
# Plotar a curva ROC
plot(roc_obj, col = "blue", lwd = 2, main = "Curva ROC - XGBoost")

# AUC: 0.9795
# O modelo tem uma capacidade alta de classificar corretamente entre as duas classes (0 e 1)



#--------------------- Coeficiente de Gini ---------------------
# Calculando o coeficiente de Gini
gini_coefficient <- 2 * auc_value - 1
print(paste("Coeficiente de Gini:", gini_coefficient))

# Coeficiente de Gini: 0.9591
# Sugere que o modelo está conseguindo classificar corretamente os clientes, separando com muita eficácia os elegíveis dos não elegíveis



#--------------------- Curva Precision-Recall ---------------------
# Obter as previsões probabilísticas no conjunto de teste
y_pred_prob <- predict(xgb_model, X_test)
# Gerar a curva Precision-Recall
pr_curve <- pr.curve(scores.class0 = y_pred_prob, weights.class0 = y_test, curve = TRUE)
# Plotar a curva Precision-Recall
plot(pr_curve)
# Exibir a área sob a curva Precision-Recall (PR AUC)
print(paste("PR AUC:", pr_curve$auc.integral))

# PR AUC: 0.9096
# Significa que o modelo tem um bom equilíbrio entre a capacidade de identificar 
# corretamente os clientes elegíveis (recall) e a precisão ao prever que eles são elegíveis.



#--------------------- Log-Loss (Perda Logarítmica) ---------------------
# Função para calcular o Log-Loss
logLoss <- function(true_labels, predicted_probs) {
  epsilon <- 1e-15  # Pequeno valor para evitar log(0)
  predicted_probs <- pmax(epsilon, pmin(1 - epsilon, predicted_probs))  # Limitar as probabilidades entre epsilon e 1 - epsilon
  -mean(true_labels * log(predicted_probs) + (1 - true_labels) * log(1 - predicted_probs))
}
# Calcular o Log-Loss
log_loss_value <- logLoss(y_test, predictions_prob)
print(paste("Log-Loss:", log_loss_value))

# Log-Loss: 0.1036
# Indica que o modelo está gerando previsões probabilísticas muito precisas e que as probabilidades associadas às classes estão bem calibradas



#--------------------- Cross-Validation ---------------------
# Preparar o conjunto de treino no formato de DMatrix
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
# Definir os parâmetros do modelo XGBoost
params <- list(
  objective = "binary:logistic",  # Problema de classificação binária
  eval_metric = "logloss"         # Métrica de avaliação durante o cross-validation
)
# Executar o cross-validation com 5-folds
cv_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 100,        # Número de iterações 
  nfold = 5,            # Número de folds para o cross-validation
  verbose = 1,          # Mostrar o progresso do cross-validation
  early_stopping_rounds = 10, # Parar se não houver melhora após 10 rounds
  maximize = FALSE       # Como estamos minimizando o logloss
)
# Melhor número de iterações (nrounds)
best_nrounds <- cv_model$best_iteration
print(paste("Melhor número de rodadas:", best_nrounds))

# Na base original, o melhor número de rodadas ficou em 28



#--------------------- Importância das Variáveis ---------------------
# Calcular a importância das variáveis no modelo original
importance_matrix_original <- xgb.importance(feature_names = colnames(X_train), model = xgb_model)
# Exibir a importância das variáveis
print(importance_matrix_original)
# Plotar as 10 variáveis mais importantes
xgb.plot.importance(importance_matrix_original, top_n = 10, main = "Top 10 Variáveis Mais Importantes - Modelo Original")

# As variáveis mais importantes para a predição são:
# 1ª: faixa_renda: 0.3363
# 2ª: possui_cartao: 0.1753
# 3ª: possui_deposito_prazo: 0.1312
# 4ª: possui_credito_repasse 0.0964
# 5ª: possui_credito_comercial 0.0594



#--------------------- Overfitting e Underfitting ---------------------
# Fazer previsões no conjunto de treino
predictions_train <- predict(xgb_model, X_train)
predicted_classes_train <- ifelse(predictions_train > 0.35, 1, 0)
accuracy_train <- mean(predicted_classes_train == y_train)
print(paste("Acurácia dataset de treino:", accuracy_train))

# Fazer previsões no conjunto de teste
predictions_test <- predict(xgb_model, X_test)
predicted_classes_test <- ifelse(predictions_test > 0.35, 1, 0)
accuracy_test <- mean(predicted_classes_test == y_test)
print(paste("Acurácia no dataset de teste:", accuracy_test))

# Acurácia dataset de treino: 0.9598
# Acurácia no dataset de teste: 0.9526
# O modelo parece estar bem ajustado e generaliza bem nos dados de teste, pois 
# não apresenta diferença relevante entre as acurácias de treino e teste



#--------------------- Análise de resíduos ---------------------
# Calcular os resíduos (diferença entre as classes reais e as probabilidades previstas)
residuals <- y_test - predictions_test
# Plotar os resíduos em um gráfico de dispersão
plot(predictions_test, residuals, 
     xlab = "Probabilidades Previstas", 
     ylab = "Resíduos (Real - Previsto)", 
     main = "Análise de Resíduos",
     pch = 20, col = "blue")
# Adicionar uma linha horizontal para destacar o zero (erro ideal)
abline(h = 0, col = "red", lwd = 2)

# Como os resíduos estão distribuídos de maneira razoavelmente aleatória e simétrica ao redor de 0, 
# sem um padrão claro, isso indica que o modelo não está cometendo erros sistemáticos.
# o gráfico sugere que o modelo está bem ajustado e que os erros são aleatórios, o que significa que 
# não há sinais claros de overfitting ou underfitting neste aspecto



#--------------------- F1-score ---------------------
# Fazer previsões no conjunto de teste
predictions_test <- predict(xgb_model, X_test)
predicted_classes_test <- ifelse(predictions_test > 0.35, 1, 0)

# Criar a matriz de confusão
confusion_matrix <- table(Predicted = predicted_classes_test, Actual = y_test)

# Extrair os valores da matriz de confusão
TP <- confusion_matrix[2, 2]  # Verdadeiros positivos
FP <- confusion_matrix[2, 1]  # Falsos positivos
FN <- confusion_matrix[1, 2]  # Falsos negativos

# Calcular precisão e recall
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

# Calcular o F1-Score
f1_score <- 2 * (precision * recall) / (precision + recall)
print(paste("Precision no dataset de teste:", precision))
print(paste("Recall no dataset de teste:", recall))
print(paste("F1-Score no dataset de teste:", f1_score))

# F1-Score no dataset de teste: 0.8155
# Indica um bom equilíbrio entre precisão e recall, especialmente considerando 
# que o valor ideal do F1-Score é 1. Esse resultado sugere que o modelo XGBoost 
# está desempenhando bem, com uma boa taxa de previsões corretas tanto para a 
# classe positiva quanto para a negativa.

