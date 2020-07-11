import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split

data = pd.read_csv("/Users/robert/Documents/Pycharm/knn/credit_data.csv")
# data = pd.read_csv("C:\\Users\\User\\Desktop\\credit_data.csv")

# Logistic regression accuracy: 93%
# we do better with knn: 98.5% !!!!!!!!
# 84%

#print(creditData.head())
#print(creditData.describe())
print(data.corr())

data.features = data[["income","age","loan"]]
data.target = data.default


data.features = preprocessing.MinMaxScaler().fit_transform(data.features) #HUGE DIFFERENCE !!!

feature_train, feature_test, target_train, target_test = train_test_split(data.features,data.target, test_size=0.3)

model = KNeighborsClassifier(n_neighbors=32)  # k value !!!
fittedModel = model.fit(feature_train, target_train)
predictions = fittedModel.predict(feature_test)

cross_valid_scores = []

for k in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,data.features,data.target,cv=10,scoring='accuracy')
    cross_valid_scores.append(scores.mean())
    

print("Optimal k with cross-validation: ", np.argmax(cross_valid_scores))    
    
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))


#           clientid    income       age      loan       LTI   default
# clientid  1.000000  0.039280 -0.030341  0.018931  0.002538 -0.020145
# income    0.039280  1.000000 -0.034984  0.441117 -0.019862  0.002284
# age      -0.030341 -0.034984  1.000000  0.006561  0.021588 -0.444765
# loan      0.018931  0.441117  0.006561  1.000000  0.847495  0.377160
# LTI       0.002538 -0.019862  0.021588  0.847495  1.000000  0.433261
# default  -0.020145  0.002284 -0.444765  0.377160  0.433261  1.000000
# Optimal k with cross-validation:  32
# [[509   2]
#  [ 13  76]]
# 0.975
