#!/usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

#
# FINANCIAL
#

# Read in file
financial_path = "../../data/financial-sampled10.csv"
financial_dataset = pd.read_csv(financial_path)

#check column types
print(financial_dataset.dtypes);

# set non-numeric types to numeric
object_columns = financial_dataset.select_dtypes(['object']).columns

# set all object columns to category type
for c in object_columns:
    financial_dataset[c] = financial_dataset[c].astype('category')

# convert category col types to numerical
financial_dataset[object_columns] = financial_dataset[object_columns].apply(lambda x: x.cat.codes)


#sns.heatmap(financial_dataset.corr(), xticklabels=True, yticklabels=True, annot=False)
print(financial_dataset.describe(include='all'))

plt.show()

#attributes with the biggest correlation val to isFraud
# step
# type
# amount
# oldbalanceOrg
# isFlaggedFraud

# creating input features and target variables
X = financial_dataset[['step','type','amount', 'oldbalanceOrg', 'isFlaggedFraud']]
y = financial_dataset[['isFraud']]

# let's normalize values
sc = StandardScaler()
X = sc.fit_transform(X)
# split to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(3, activation='relu', kernel_initializer='random_normal', input_dim=5))
#Second  Hidden Layer
classifier.add(Dense(2, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(5, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(2, activation='relu', kernel_initializer='random_normal'))

# As this is a binary classification problem we will use sigmoid as the activation function.
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=10)

y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)
