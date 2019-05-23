#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

#
# Amazon review
#


ar_train_path = "../../data/amazon-review/amazon_review_ID.shuf.lrn.csv"
ar_train_dataset = pd.read_csv(ar_train_path)

#check column types

#print(ar_train_dataset['Class'].describe())

#print(ar_train_dataset.dtypes);

target_values_string = ar_train_dataset.Class.unique().tolist()

ar_train_dataset['Class'] = ar_train_dataset['Class'].astype('category')
ar_train_dataset['Class'] = ar_train_dataset['Class'].cat.codes

# creating input features and target variables
X= ar_train_dataset.iloc[:,1:10001]
y= ar_train_dataset.iloc[:,10001:10002]

target_values_numeric = y.Class.unique().tolist()

target_values_mapping = dict(zip(target_values_numeric, target_values_string))

# # let's normalize values
sc = StandardScaler()
X = sc.fit_transform(X)

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(500, activation='relu', kernel_initializer='random_normal', input_dim=10000))
#Second  Hidden Layer
classifier.add(Dense(400, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(300, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(150, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(75, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(50, activation='softmax', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='sparse_categorical_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X, y, batch_size=2, epochs=15)

ar_test_path = "../../data/amazon-review/amazon_review_ID.shuf.tes.csv"
ar_test_dataset = pd.read_csv(ar_test_path)
ids = ar_test_dataset['ID'].tolist()

# # creating input features and target variables
# X_test = bc_test_dataset[bc_features]
X_test= ar_test_dataset.iloc[:,1:10001]
X_test = sc.fit_transform(X_test)


f = open("../../results/peter-nn/amazon-reviews/sub-ar.csv", "w")
f.write('ID,"Class"\n')
y_pred=classifier.predict(X_test)
for i in range(len(y_pred)):
    val = target_values_mapping.get(int(np.argmax(y_pred[i])))
    f.write(str(ids[i]) +','+ str(val) + '\n')
f.close()
