import numpy as np
import pandas as pd

iris = pd.read_csv('Iris.csv')

iris.head()

iris.info()

X = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
X.head()

y = iris['Species']
y.head()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
model1 = abc.fit(X_train, y_train)
y_pred = model1.predict(X_test)


from sklearn.metrics import accuracy_score
print("AdaBoost Classifier Model Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc=SVC(probability=True, kernel='linear')
abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1, random_state=0)
model2 = abc.fit(X_train, y_train)

y_pred = model2.predict(X_test)

print("Model Accuracy with SVC Base Estimator:",accuracy_score(y_test, y_pred))
