#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

#
# CRIME MAPPING
#

# Read in file
crime_mapping_path = "../../data/crime-mapping.csv"
crime_mapping_dataset = pd.read_csv(crime_mapping_path, sep = ';')

#check column types
print(crime_mapping_dataset.dtypes);

# set non-numeric types to numeric
object_columns = crime_mapping_dataset.select_dtypes(['object']).columns

# set all object columns to category type
for c in object_columns:
    crime_mapping_dataset[c] = crime_mapping_dataset[c].astype('category')

# convert category col types to numerical
crime_mapping_dataset[object_columns] = crime_mapping_dataset[object_columns].apply(lambda x: x.cat.codes)

sns.heatmap(crime_mapping_dataset.corr(), xticklabels=True, yticklabels=True, annot=False)
plt.show()

#attributes with the biggest correlation val to phxcommunity
# apartment_complex
# activity_date
# phxrecordstatus
# phxstatus
# domestic

# creating input features and target variables
X = crime_mapping_dataset[['apartment_complex','activity_date','phxrecordstatus', 'phxstatus', 'domestic']]
y = crime_mapping_dataset[['phxcommunity']]

# let's normalize values
sc = StandardScaler()
X = sc.fit_transform(X)
# split to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(3, activation='relu', kernel_initializer='random_normal', input_dim=5))
#Second  Hidden Layer
classifier.add(Dense(3, activation='relu', kernel_initializer='random_normal'))
# As this is a binary classification problem we will use sigmoid as the activation function.
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)
#1248/(1248+225) = 0.84725050916

