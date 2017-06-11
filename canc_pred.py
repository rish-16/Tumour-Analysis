import csv
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

def len_dataframe():
	return len(cancer['feature_names'])

#----------------------------------------------------------------------------------------------------------------

def cvt_into_dataframe():
	columns = cancer['feature_names']
	data = cancer['data']
	target = np.array(cancer['target'])

	df1 = pd.DataFrame(data)
	df2 = pd.DataFrame(target)

	df = pd.concat([df1, df2], ignore_index=True, axis=1)

	df.columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
	'mean smoothness', 'mean compactness', 'mean concavity',
	'mean concave points', 'mean symmetry', 'mean fractal dimension',
	'radius error', 'texture error', 'perimeter error', 'area error',
	'smoothness error', 'compactness error', 'concavity error',
	'concave points error', 'symmetry error', 'fractal dimension error',
	'worst radius', 'worst texture', 'worst perimeter', 'worst area',
	'worst smoothness', 'worst compactness', 'worst concavity',
	'worst concave points', 'worst symmetry', 'worst fractal dimension',
	'target']

	return (df)

#----------------------------------------------------------------------------------------------------------------

# Checks for malignant and benign tumour ratio
def check_mb_ratio():
	cancerdf = cvt_into_dataframe()

	t_list = cancerdf.iloc[:,-1]

	# target = pd.Series([0,1], index=['malignant','benign'], dtype=np.float64)
	target = pd.Series(np.float64([0,1]), index=['malignant','benign'])

	return (target)

#----------------------------------------------------------------------------------------------------------------

def get_x_y():
	cancerdf = cvt_into_dataframe()

	X = cancerdf.iloc[:, 0:30]
	y = cancerdf.iloc[:, -1]

	return (X, y)

#----------------------------------------------------------------------------------------------------------------

def get_train_test_data():
	X, y = get_x_y()

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

	return (X_train, X_test, y_train, y_test)

#----------------------------------------------------------------------------------------------------------------

def get_train_test_shape():
	X, y = get_x_y()

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

	return (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#----------------------------------------------------------------------------------------------------------------

def create_knn():
	X_train, X_test, y_train, y_test = get_train_test_data()

	knn = KNeighborsClassifier(n_neighbors=1)

	knn.fit(X_train, y_train)

	return (knn)

#----------------------------------------------------------------------------------------------------------------

def mean_prediction():

	cancerdf = cvt_into_dataframe()
	means = cancerdf.mean()[:-1].values.reshape(1, -1)

	knn = create_knn()

	prediction = knn.predict(means)

	return (prediction)

#----------------------------------------------------------------------------------------------------------------

def get_prediction():

	X_train, X_test, y_train, y_test = get_train_test_data()
	trained_knn = create_knn()

	prediction = trained_knn.predict(X_test)

	return (prediction)

#----------------------------------------------------------------------------------------------------------------

def test_accuracy():

	X_train, X_test, y_train, y_test = get_train_test_data()
	trained_knn = create_knn()

	tested_knn = trained_knn.predict(X_test)
	accuracy = trained_knn.score(X_test, y_test)

	return accuracy

#----------------------------------------------------------------------------------------------------------------

len_dataframe = len_dataframe()

dataframe = cvt_into_dataframe()

target_list = check_mb_ratio()

X, y = get_x_y()

X_train, X_test, y_train, y_test = get_train_test_data()

X_trainShape, X_testShape, y_trainShape, y_testShape = get_train_test_shape()

my_knn = create_knn()

mean_pred = mean_prediction()

test_prediction = get_prediction()

knn_accuracy = test_accuracy()

#----------------------------------------------------------------------------------------------------------------

print (len_dataframe)

print (dataframe)

print (target_list)

print (X, y)

print (X_train, X_test, y_train, y_test)

print (X_trainShape, X_testShape, y_trainShape, y_testShape)

print (my_knn)

print ("Mean Results: {}".format(mean_pred))

print ("Results: {}".format(test_prediction))

print ("Testing accuracy: {}".format(knn_accuracy))
