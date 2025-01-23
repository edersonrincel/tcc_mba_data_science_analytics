# Pacotes necessários
pacotes <- c("MLmetrics", "PRROC", "caret", "pROC")

if (sum(as.numeric(!pacotes %in% installed.packages())) != 0) {
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for (i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
  }
  sapply(pacotes, require, character = T)
} else {
  sapply(pacotes, require, character = T)
}

# Carrega dados gerados pelo script "2.tratamento.R"
data <- read.csv("2.base_normalizada.csv")

# Separar variáveis da variável target
X <- data[, -ncol(data)]  # Todas as colunas exceto a última
y <- data$possui_cartao_black  # Variável target

# Separar dados entre treino e teste (80% treino, 20% teste)
set.seed(123)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Ajustar o modelo de regressão logística
log_model <- glm(as.factor(y_train) ~ ., data = X_train, family = binomial)

# Fazer predições probabilísticas na base de teste
predictions_prob <- predict(log_model, newdata = X_test, type = "response")

# Converter as probabilidades em classes (threshold = 0.5)
predicted_classes <- ifelse(predictions_prob > 0.5, 1, 0)

# Aferição da acurácia
accuracy <- mean(predicted_classes == y_test)
print(paste("Acurácia:", accuracy))

# Avaliação do modelo

# Matriz de confusão
confusion_matrix <- confusionMatrix(as.factor(predicted_classes), as.factor(y_test))
print(confusion_matrix)

# AUC-ROC
roc_obj <- roc(y_test, predictions_prob)
auc_value <- auc(roc_obj)
print(paste("AUC:", auc_value))
plot(roc_obj, col = "blue", lwd = 2, main = "Curva ROC - Regressão Logística")

# Coeficiente de Gini
gini_coefficient <- 2 * auc_value - 1
print(paste("Coeficiente de Gini:", gini_coefficient))

# Log-Loss (Perda Logarítmica)
logLoss <- function(true_labels, predicted_probs) {
  epsilon <- 1e-15
  predicted_probs <- pmax(epsilon, pmin(1 - epsilon, predicted_probs))
  -mean(true_labels * log(predicted_probs) + (1 - true_labels) * log(1 - predicted_probs))
}
log_loss_value <- logLoss(y_test, predictions_prob)
print(paste("Log-Loss:", log_loss_value))

# Curva Precision-Recall
pr_curve <- pr.curve(scores.class0 = predictions_prob, weights.class0 = y_test, curve = TRUE)
plot(pr_curve)
print(paste("PR AUC:", pr_curve$auc.integral))
