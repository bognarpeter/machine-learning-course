breast.cancer <- read.csv("/Users/stefanpuhalo/TU Master/Machine Learning/datasets/184702-tu-ml-ss-19-breast-cancer/breast-cancer-diagnostic.shuf.lrn.csv", 
                          sep = ",",
                          header = TRUE)
breast.cancer$ID <- NULL
breast.cancer.exp <- breast.cancer
breast.cancer.exp$class <- NULL
breast.cancer.train <- breast.cancer.exp[1:(nrow(breast.cancer.exp) * 0.8),]
breast.cancer.test <- breast.cancer.exp[(nrow(breast.cancer.exp) * 0.8 + 1):nrow(breast.cancer.exp),]
library(class)

start_time_1 <- Sys.time()
breast_cancer_pred_1 <- knn(breast.cancer.train, breast.cancer.test, breast.cancer[1:(nrow(breast.cancer.exp) * 0.8),1], k=1)
end_time_1 <- Sys.time()
duration_1 <- end_time_1 - start_time_1
tab_1 <- table(breast_cancer_pred_1, breast.cancer[(nrow(breast.cancer.exp) * 0.8 + 1):nrow(breast.cancer.exp),1])
acc_1 <- sum(diag(tab_1)/sum(tab_1))

start_time_3 <- Sys.time()
breast_cancer_pred_3 <- knn(breast.cancer.train, breast.cancer.test, breast.cancer[1:(nrow(breast.cancer.exp) * 0.8),1], k=3)
end_time_3 <- Sys.time()
duration_3 <- end_time_3 - start_time_3
tab_3 <- table(breast_cancer_pred_3, breast.cancer[(nrow(breast.cancer.exp) * 0.8 + 1):nrow(breast.cancer.exp),1])
acc_3 <- sum(diag(tab_3)/sum(tab_3))

start_time_5 <- Sys.time()
breast_cancer_pred_5 <- knn(breast.cancer.train, breast.cancer.test, breast.cancer[1:(nrow(breast.cancer.exp) * 0.8),1], k=5)
end_time_5 <- Sys.time()
duration_5 <- end_time_5 - start_time_5
tab_5 <- table(breast_cancer_pred_5, breast.cancer[(nrow(breast.cancer.exp) * 0.8 + 1):nrow(breast.cancer.exp),1])
acc_5 <- sum(diag(tab_5)/sum(tab_5))

start_time_7 <- Sys.time()
breast_cancer_pred_7 <- knn(breast.cancer.train, breast.cancer.test, breast.cancer[1:(nrow(breast.cancer.exp) * 0.8),1], k=7)
end_time_7 <- Sys.time()
duration_7 <- end_time_7 - start_time_7
tab_7 <- table(breast_cancer_pred_7, breast.cancer[(nrow(breast.cancer.exp) * 0.8 + 1):nrow(breast.cancer.exp),1])
acc_7 <- sum(diag(tab_7)/sum(tab_7))

start_time_16 <- Sys.time()
breast_cancer_pred_16 <- knn(breast.cancer.train, breast.cancer.test, breast.cancer[1:(nrow(breast.cancer.exp) * 0.8),1], k=16)
end_time_16 <- Sys.time()
duration_16 <- end_time_16 - start_time_16
tab_16 <- table(breast_cancer_pred_16, breast.cancer[(nrow(breast.cancer.exp) * 0.8 + 1):nrow(breast.cancer.exp),1])
acc_16 <- sum(diag(tab_16)/sum(tab_16))

start_time_17 <- Sys.time()
breast_cancer_pred_17 <- knn(breast.cancer.train, breast.cancer.test, breast.cancer[1:(nrow(breast.cancer.exp) * 0.8),1], k=17)
end_time_17 <- Sys.time()
duration_17 <- end_time_17 - start_time_17
tab_17 <- table(breast_cancer_pred_17, breast.cancer[(nrow(breast.cancer.exp) * 0.8 + 1):nrow(breast.cancer.exp),1])
acc_17 <- sum(diag(tab_17)/sum(tab_17))

start_time_25 <- Sys.time()
breast_cancer_pred_25 <- knn(breast.cancer.train, breast.cancer.test, breast.cancer[1:(nrow(breast.cancer.exp) * 0.8),1], k=25)
end_time_25 <- Sys.time()
duration_25 <- end_time_25 - start_time_25
tab_25 <- table(breast_cancer_pred_25, breast.cancer[(nrow(breast.cancer.exp) * 0.8 + 1):nrow(breast.cancer.exp),1])
acc_25 <- sum(diag(tab_25)/sum(tab_25))

start_time_35 <- Sys.time()
breast_cancer_pred_35 <- knn(breast.cancer.train, breast.cancer.test, breast.cancer[1:(nrow(breast.cancer.exp) * 0.8),1], k=35)
end_time_35 <- Sys.time()
duration_35 <- end_time_35 - start_time_35
tab_35 <- table(breast_cancer_pred_35, breast.cancer[(nrow(breast.cancer.exp) * 0.8 + 1):nrow(breast.cancer.exp),1])
acc_35 <- sum(diag(tab_35)/sum(tab_35))

start_time_227 <- Sys.time()
breast_cancer_pred_227 <- knn(breast.cancer.train, breast.cancer.test, breast.cancer[1:(nrow(breast.cancer.exp) * 0.8),1], k=227)
end_time_227 <- Sys.time()
duration_227 <- end_time_227 - start_time_227
tab_227<- table(breast_cancer_pred_227, breast.cancer[(nrow(breast.cancer.exp) * 0.8 + 1):nrow(breast.cancer.exp),1])
acc_227 <- sum(diag(tab_227)/sum(tab_227))

acc <- c(acc_1, acc_3, acc_5, acc_7, acc_16, acc_17, acc_25, acc_35, acc_227)
acc <- acc * 100
k <- c(1,3,5,7,16,17,25,35,227)
duration <- c(duration_1, duration_3, duration_5, duration_7, duration_16, duration_17, duration_25, duration_35, duration_227)
df_unnorm <- data.frame(k,acc,duration)
plot(k,acc, main="KNN - Accuracy", type="o", ylab = "Accuracy", col="red")
plot(k,duration, main="KNN - Process Duration", type="o", ylab = "duration (s)", col="red")
