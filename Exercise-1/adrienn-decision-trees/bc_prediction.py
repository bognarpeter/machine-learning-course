#   --
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
#ID,class,radiusMean, textureMean, perimeterMean, areaMean, smoothnessMean, compactnessMean, concavityMean, concavePointsMean, symmetryMean
col_names = ['ID', 'label',  'radiusMean', 'textureMean', 'perimeterMean','areaMean', 'smoothnessMean', 'compactnessMean', 'concavityMean','concavePointsMean', 'symmetryMean', 'fractalDimensionMean', 'radiusStdErr', 'textureStdErr','perimeterStdErr', 'areaStdErr', 'smoothnessStdErr', 'compactnessStdErr', 'concavityStdErr', 'concavePointsStdErr', 'symmetryStdErr', 'fractalDimensionStdErr', 'radiusWorst', 'textureWorst','perimeterWorst', 'areaWorst', 'smoothnessWorst', 'compactnessWorst', ' concavityWorst', 'concavePointsWorst', 'symmetryWorst', 'fractalDimensionWorst ']
# load dataset
pima = pd.read_csv("bc_train.csv")
pima.columns = col_names

col_name2 = ['ID',  'radiusMean', 'textureMean', 'perimeterMean','areaMean', 'smoothnessMean', 'compactnessMean', 'concavityMean','concavePointsMean', 'symmetryMean', 'fractalDimensionMean', 'radiusStdErr', 'textureStdErr','perimeterStdErr', 'areaStdErr', 'smoothnessStdErr', 'compactnessStdErr', 'concavityStdErr', 'concavePointsStdErr', 'symmetryStdErr', 'fractalDimensionStdErr', 'radiusWorst', 'textureWorst','perimeterWorst', 'areaWorst', 'smoothnessWorst', 'compactnessWorst', ' concavityWorst', 'concavePointsWorst', 'symmetryWorst', 'fractalDimensionWorst ']
pima2 = pd.read_csv("bc_test.csv")
pima2.columns = col_name2

feature_cols2 = [ 'radiusMean', 'textureMean', 'perimeterMean','areaMean', 'smoothnessMean', 'compactnessMean', 'concavityMean','concavePointsMean', 'symmetryMean']
X2 = pima2[feature_cols2] # test set for new unseen data



#split dataset in features and target variable
# do not include the error values in the training
feature_cols = [ 'radiusMean', 'textureMean', 'perimeterMean','areaMean', 'smoothnessMean', 'compactnessMean', 'concavityMean','concavePointsMean', 'symmetryMean']
X = pima[feature_cols] # Features
y = pima.label # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)




# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion = 'entropy')

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X2)

print(y_pred)



from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('figure1.png')
Image(graph.create_png())

 