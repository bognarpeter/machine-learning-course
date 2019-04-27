amazon <- read.csv("/Users/stefanpuhalo/TU Master/Machine Learning/datasets/Amazon_Review_Data/amazon_review_ID.shuf.lrn.csv", 
                          sep = ",",
                          header = TRUE)
amazon$ID <- NULL
data_norm <- function(x) { ((x - min(x))/(max(x) - min(x))) }
amazon.norm <- amazon
amazon.norm$class <- NULL
amazon.norm <- as.data.frame(lapply(amazon.norm, data_norm))
amazon.train <- amazon.norm[1:(nrow(amazon.norm) * 0.8),1:10000]
amazon.test <- amazon.norm[(nrow(amazon.norm) * 0.8 + 1):nrow(amazon.norm),1:10000]
library(class)
#amazon_pred <- knn(amazon.train, amazon.test, amazon[1:(nrow(amazon) * 0.8),10001], k=27)
#tab <- table(amazon_pred, amazon[(nrow(amazon) * 0.8 + 1):nrow(amazon),10001])
#sum(diag(tab)/sum(tab))

start_time_1 <- Sys.time()
amazon_pred_1 <- knn(amazon.train, amazon.test, amazon[1:(nrow(amazon.norm) * 0.8),10001], k=1)
end_time_1 <- Sys.time()
duration_1 <- end_time_1 - start_time_1
tab_1 <- table(amazon_pred_1, amazon[(nrow(amazon.norm) * 0.8 + 1):nrow(amazon.norm),10001])
acc_1 <- sum(diag(tab_1)/sum(tab_1))

start_time_3 <- Sys.time()
amazon_pred_3 <- knn(amazon.train, amazon.test, amazon[1:(nrow(amazon.norm) * 0.8),10001], k=3)
end_time_3 <- Sys.time()
duration_3 <- end_time_3 - start_time_3
tab_3 <- table(amazon_pred_3, amazon[(nrow(amazon.norm) * 0.8 + 1):nrow(amazon.norm),10001])
acc_3 <- sum(diag(tab_3)/sum(tab_3))

start_time_5 <- Sys.time()
amazon_pred_5 <- knn(amazon.train, amazon.test, amazon[1:(nrow(amazon.norm) * 0.8),10001], k=5)
end_time_5 <- Sys.time()
duration_5 <- end_time_5 - start_time_5
tab_5 <- table(amazon_pred_5, amazon[(nrow(amazon.norm) * 0.8 + 1):nrow(amazon.norm),10001])
acc_5 <- sum(diag(tab_5)/sum(tab_5))

start_time_7 <- Sys.time()
amazon_pred_7 <- knn(amazon.train, amazon.test, amazon[1:(nrow(amazon.norm) * 0.8),10001], k=7)
end_time_7 <- Sys.time()
duration_7 <- end_time_7 - start_time_7
tab_7 <- table(amazon_pred_7, amazon[(nrow(amazon.norm) * 0.8 + 1):nrow(amazon.norm),10001])
acc_7 <- sum(diag(tab_7)/sum(tab_7))

start_time_17 <- Sys.time()
amazon_pred_17 <- knn(amazon.train, amazon.test, amazon[1:(nrow(amazon.norm) * 0.8),10001], k=17)
end_time_17 <- Sys.time()
duration_17 <- end_time_17 - start_time_17
tab_17 <- table(amazon_pred_17, amazon[(nrow(amazon.norm) * 0.8 + 1):nrow(amazon.norm),10001])
acc_17 <- sum(diag(tab_17)/sum(tab_17))

start_time_27 <- Sys.time()
amazon_pred_27 <- knn(amazon.train, amazon.test, amazon[1:(nrow(amazon.norm) * 0.8),10001], k=27)
end_time_27 <- Sys.time()
duration_27 <- end_time_27 - start_time_27
tab_27 <- table(amazon_pred_27, amazon[(nrow(amazon.norm) * 0.8 + 1):nrow(amazon.norm),10001])
acc_27 <- sum(diag(tab_27)/sum(tab_27))

start_time_35 <- Sys.time()
amazon_pred_35 <- knn(amazon.train, amazon.test, amazon[1:(nrow(amazon.norm) * 0.8),10001], k=35)
end_time_35 <- Sys.time()
duration_35 <- end_time_35 - start_time_35
tab_35 <- table(amazon_pred_35, amazon[(nrow(amazon.norm) * 0.8 + 1):nrow(amazon.norm),10001])
acc_35 <- sum(diag(tab_35)/sum(tab_35))

start_time_50 <- Sys.time()
amazon_pred_50 <- knn(amazon.train, amazon.test, amazon[1:(nrow(amazon.norm) * 0.8),10001], k=50)
end_time_50 <- Sys.time()
duration_50 <- end_time_50 - start_time_50
tab_50<- table(amazon_pred_50, amazon[(nrow(amazon.norm) * 0.8 + 1):nrow(amazon.norm),10001])
acc_50 <- sum(diag(tab_50)/sum(tab_50))

start_time_69 <- Sys.time()
amazon_pred_69 <- knn(amazon.train, amazon.test, amazon[1:(nrow(amazon.norm) * 0.8),10001], k=69)
end_time_69 <- Sys.time()
duration_69 <- end_time_69 - start_time_69
tab_69<- table(amazon_pred_69, amazon[(nrow(amazon.norm) * 0.8 + 1):nrow(amazon.norm),10001])
acc_69 <- sum(diag(tab_69)/sum(tab_69))

start_time_250 <- Sys.time()
amazon_pred_250 <- knn(amazon.train, amazon.test, amazon[1:(nrow(amazon.norm) * 0.8),10001], k=250)
end_time_250 <- Sys.time()
duration_250 <- end_time_250 - start_time_250
tab_250<- table(amazon_pred_250, amazon[(nrow(amazon.norm) * 0.8 + 1):nrow(amazon.norm),10001])
acc_250 <- sum(diag(tab_250)/sum(tab_250))


acc <- c(acc_1, acc_3, acc_5, acc_7, acc_17, acc_27, acc_35, acc_50, acc_69, acc_250)
acc <- acc * 100
k <- c(1,3,5,7,17,27,35,50,69,250)
duration <- c(duration_1, duration_3, duration_5, duration_7, 
              duration_17, duration_27, duration_35, duration_50, duration_69, duration_250)
df <- data.frame(k,acc,duration)
plot(k,acc, main="KNN - Accuracy", type="o", ylab = "Accuracy", col="red")
plot(k,duration, main="KNN - Process Duration", type="o", ylab = "duration (s)", col="red")
