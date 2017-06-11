import numpy as np
from sklearn import cross_validation, preprocessing, neighbors, svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1), dtype=int)
y = np.array(df['class'], dtype=int)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print (accuracy)

eg_mea = np.array([[4,2,1,1,1,2,3,2])

eg_mea = eg_mea.reshape(len(eg_mea),-1)

prediction = clf.predict(eg_mea)
print(prediction)
