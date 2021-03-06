#   --
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['ccat', 'district', 'beat', 'mapref', 'label', 'domestic']
# load dataset
pima = pd.read_csv("crimemap_cleared.csv")

pima.columns = col_names

#pima = pd.read_csv("diabetes.csv", header=None, names=col_names)
#pima.head()

#split dataset in features and target variable
feature_cols = ['ccat', 'district', 'beat', 'mapref', 'domestic']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



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

 #Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=10) #maxdepth=3Accuracy: 0.8610335195530726 ,Accuracy: 0.8589385474860335

 #with max_depth=10 : Accuracy: 0.8603351955307262
 #Accuracy: 0.8652234636871509
 #[Finished in 40.9s]


# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))









from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('figure2.png')
Image(graph.create_png())