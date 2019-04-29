#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense



#
# Breast-cancer
#

bc_train_path = "../../data/breast-cancer/breast-cancer-diagnostic.shuf.lrn.csv"
bc_train_dataset = pd.read_csv(bc_train_path)

#check column types
print(bc_train_dataset.dtypes);

bc_train_dataset['class'] = bc_train_dataset['class'].astype('category')
bc_train_dataset['class'] = bc_train_dataset['class'].cat.codes

sns.heatmap(bc_train_dataset.corr(), xticklabels=True, yticklabels=True, annot=False)
plt.show()

bc_features = ['radiusMean','perimeterMean','areaMean', 'compactnessMean', 'concavityMean', 'concavePointsMean', 'radiusStdErr', 'perimeterStdErr', 'areaStdErr', 'radiusWorst', 'perimeterWorst', 'areaWorst', 'compactnessWorst', 'concavityWorst', 'concavePointsWorst','symmetryWorst']
# # creating input features and target variables
# X = bc_train_dataset[bc_features]
# y = bc_train_dataset[['class']]
X= bc_train_dataset.iloc[:,2:32]
y= bc_train_dataset.iloc[:,1:2]

# # let's normalize values
sc = StandardScaler()
X = sc.fit_transform(X)

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal', input_dim=30))
#Second  Hidden Layer
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))

# As this is a binary classification problem we will use sigmoid as the activation function.
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X, y, batch_size=4, epochs=100)


bc_test_path = "../../data/breast-cancer/breast-cancer-diagnostic.shuf.tes.csv"
bc_test_dataset = pd.read_csv(bc_test_path)
ids = bc_test_dataset['ID'].tolist()

# # creating input features and target variables
# X_test = bc_test_dataset[bc_features]
X_test= bc_test_dataset.iloc[:,1:31]
X_test = sc.fit_transform(X_test)

y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)

for i in range(len(y_pred)):
    if str(y_pred[i][0]) == 'True':
        print(str(ids[i]) + ',M')
    elif str(y_pred[i][0]) is 'False':
        print(str(ids[i]) + ',B')


