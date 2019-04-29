fraud <- read.csv("/Users/stefanpuhalo/TU Master/Machine Learning/datasets/financial-sampled10.csv", 
                  sep = ",",
                  header = TRUE)
fraud.subset <- fraud[c('type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest')]
fraud.subset.cat <- fraud.subset
fraud.subset.cat$type <- as.numeric(factor(fraud.subset$type))
data_norm <- function(x) { ((x - min(x))/(max(x) - min(x))) }
fraud.subset.norm <- as.data.frame(lapply(fraud.subset.cat, data_norm))
fraud_train <- fraud.subset.norm[1:(nrow(fraud.subset.norm) * 0.8),]
fraud_test <- fraud.subset.norm[(nrow(fraud.subset.norm) * 0.8 + 1):(nrow(fraud.subset.norm)),]
library(class)
##fraud_pred <- knn(fraud_train, fraud_test, fraud[1:(nrow(fraud.subset.norm) * 0.8),10], k=1)
##table(fraud_pred, fraud[(nrow(fraud.subset.norm) * 0.8 + 1):(nrow(fraud.subset.norm)),10])

start_time_1 <- Sys.time()
fraud_pred_1 <- knn(fraud_train, fraud_test, fraud[1:(nrow(fraud.subset.norm) * 0.8),10], k=1)
end_time_1 <- Sys.time()
duration_1 <- end_time_1 - start_time_1
tab_1 <- table(fraud_pred_1, fraud[(nrow(fraud.subset.norm) * 0.8 + 1):nrow(fraud.subset.norm),10])
acc_1 <- sum(diag(tab_1)/sum(tab_1))

start_time_3 <- Sys.time()
fraud_pred_3 <- knn(fraud_train, fraud_test, fraud[1:(nrow(fraud.subset.norm) * 0.8),10], k=3)
end_time_3 <- Sys.time()
duration_3 <- end_time_3 - start_time_3
tab_3 <- table(fraud_pred_3, fraud[(nrow(fraud.subset.norm) * 0.8 + 1):nrow(fraud.subset.norm),10])
acc_3 <- sum(diag(tab_3)/sum(tab_3))

start_time_5 <- Sys.time()
fraud_pred_5 <- knn(fraud_train, fraud_test, fraud[1:(nrow(fraud.subset.norm) * 0.8),10], k=5)
end_time_5 <- Sys.time()
duration_5 <- end_time_5 - start_time_5
tab_5 <- table(fraud_pred_5, fraud[(nrow(fraud.subset.norm) * 0.8 + 1):nrow(fraud.subset.norm),10])
acc_5 <- sum(diag(tab_5)/sum(tab_5))


acc <- c(acc_1, acc_3, acc_5)
acc <- acc * 100
k <- c(1,3,5)
duration <- c(duration_1, duration_3, duration_5)
df <- data.frame(k,acc,duration)
plot(k,acc, main="KNN - Accuracy", type="o", ylab = "Accuracy", col="red")
plot(k,duration, main="KNN - Process Duration", type="o", ylab = "duration (m)", col="red")