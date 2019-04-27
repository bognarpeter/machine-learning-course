crime <- read.table("/Users/stefanpuhalo/TU Master/Machine Learning/datasets/crime-mapping.csv",
                    sep = ";", header = TRUE)
crime.subset <- crime[c('crime_category','district','beat_number','map_reference','domestic')]
crime.subset.cat <- crime.subset
crime.subset.cat$district <- as.numeric(factor(crime.subset$district))
crime.subset.cat$crime_category <- as.numeric(factor(crime.subset$crime_category))
crime.subset.cat$beat_number <- as.numeric(factor(crime.subset$beat_number))
crime.subset.cat$map_reference <- as.numeric(factor(crime.subset$map_reference))
crime.subset.cat$domestic <- as.numeric(factor(crime.subset$domestic))
data_norm <- function(x) { ((x - min(x))/(max(x) - min(x))) }
crime.subset.norm <- as.data.frame(lapply(crime.subset.cat, data_norm))
crime_train <- crime.subset.norm[1:(nrow(crime.subset.norm) * 0.8),]
crime_test <- crime.subset.norm[(nrow(crime.subset.norm) * 0.8 + 1):nrow(crime.subset.norm),]
library(class)

start_time_1 <- Sys.time()
crime_mapping_pred_1 <- knn(crime_train, crime_test, crime[1:(nrow(crime.subset.norm) * 0.8),21], k=1)
end_time_1 <- Sys.time()
duration_1 <- end_time_1 - start_time_1
tab_1 <- table(crime_mapping_pred_1, crime[(nrow(crime.subset.norm) * 0.8 + 1):nrow(crime.subset.norm),21])
acc_1 <- sum(diag(tab_1)/sum(tab_1))

start_time_3 <- Sys.time()
crime_mapping_pred_3 <- knn(crime_train, crime_test, crime[1:(nrow(crime.subset.norm) * 0.8),21], k=3)
end_time_3 <- Sys.time()
duration_3 <- end_time_3 - start_time_3
tab_3 <- table(crime_mapping_pred_3, crime[(nrow(crime.subset.norm) * 0.8 + 1):nrow(crime.subset.norm),21])
acc_3 <- sum(diag(tab_3)/sum(tab_3))

start_time_5 <- Sys.time()
crime_mapping_pred_5 <- knn(crime_train, crime_test, crime[1:(nrow(crime.subset.norm) * 0.8),21], k=5)
end_time_5 <- Sys.time()
duration_5 <- end_time_5 - start_time_5
tab_5 <- table(crime_mapping_pred_5, crime[(nrow(crime.subset.norm) * 0.8 + 1):nrow(crime.subset.norm),21])
acc_5 <- sum(diag(tab_5)/sum(tab_5))

start_time_7 <- Sys.time()
crime_mapping_pred_7 <- knn(crime_train, crime_test, crime[1:(nrow(crime.subset.norm) * 0.8),21], k=7)
end_time_7 <- Sys.time()
duration_7 <- end_time_7 - start_time_7
tab_7 <- table(crime_mapping_pred_7, crime[(nrow(crime.subset.norm) * 0.8 + 1):nrow(crime.subset.norm),21])
acc_7 <- sum(diag(tab_7)/sum(tab_7))

start_time_17 <- Sys.time()
crime_mapping_pred_17 <- knn(crime_train, crime_test, crime[1:(nrow(crime.subset.norm) * 0.8),21], k=17)
end_time_17 <- Sys.time()
duration_17 <- end_time_17 - start_time_17
tab_17 <- table(crime_mapping_pred_17, crime[(nrow(crime.subset.norm) * 0.8 + 1):nrow(crime.subset.norm),21])
acc_17 <- sum(diag(tab_17)/sum(tab_17))

start_time_25 <- Sys.time()
crime_mapping_pred_25 <- knn(crime_train, crime_test, crime[1:(nrow(crime.subset.norm) * 0.8),21], k=25)
end_time_25 <- Sys.time()
duration_25 <- end_time_25 - start_time_25
tab_25 <- table(crime_mapping_pred_25, crime[(nrow(crime.subset.norm) * 0.8 + 1):nrow(crime.subset.norm),21])
acc_25 <- sum(diag(tab_25)/sum(tab_25))

start_time_35 <- Sys.time()
crime_mapping_pred_35 <- knn(crime_train, crime_test, crime[1:(nrow(crime.subset.norm) * 0.8),21], k=35)
end_time_35 <- Sys.time()
duration_35 <- end_time_35 - start_time_35
tab_35 <- table(crime_mapping_pred_35, crime[(nrow(crime.subset.norm) * 0.8 + 1):nrow(crime.subset.norm),21])
acc_35 <- sum(diag(tab_35)/sum(tab_35))

start_time_50 <- Sys.time()
crime_mapping_pred_50 <- knn(crime_train, crime_test, crime[1:(nrow(crime.subset.norm) * 0.8),21], k=50)
end_time_50 <- Sys.time()
duration_50 <- end_time_50 - start_time_50
tab_50<- table(crime_mapping_pred_50, crime[(nrow(crime.subset.norm) * 0.8 + 1):nrow(crime.subset.norm),21])
acc_50 <- sum(diag(tab_50)/sum(tab_50))

start_time_69 <- Sys.time()
crime_mapping_pred_69 <- knn(crime_train, crime_test, crime[1:(nrow(crime.subset.norm) * 0.8),21], k=69)
end_time_69 <- Sys.time()
duration_69 <- end_time_69 - start_time_69
tab_69<- table(crime_mapping_pred_69, crime[(nrow(crime.subset.norm) * 0.8 + 1):nrow(crime.subset.norm),21])
acc_69 <- sum(diag(tab_69)/sum(tab_69))

start_time_250 <- Sys.time()
crime_mapping_pred_250 <- knn(crime_train, crime_test, crime[1:(nrow(crime.subset.norm) * 0.8),21], k=250)
end_time_250 <- Sys.time()
duration_250 <- end_time_250 - start_time_250
tab_250<- table(crime_mapping_pred_250, crime[(nrow(crime.subset.norm) * 0.8 + 1):nrow(crime.subset.norm),21])
acc_250 <- sum(diag(tab_250)/sum(tab_250))


acc <- c(acc_1, acc_3, acc_5, acc_7, acc_17, acc_25, acc_35, acc_50, acc_69, acc_250)
acc <- acc * 100
k <- c(1,3,5,7,17,25,35,50,69,250)
duration <- c(duration_1, duration_3, duration_5, duration_7, 
              duration_17, duration_25, duration_35, duration_50, duration_69, duration_250)
df <- data.frame(k,acc,duration)
plot(k,acc, main="KNN - Accuracy", type="o", ylab = "Accuracy", col="red")
plot(k,duration, main="KNN - Process Duration", type="o", ylab = "duration (s)", col="red")

