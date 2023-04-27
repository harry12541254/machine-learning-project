"""
File: titanic_level2.py
Name: 
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle website. Hyper-parameters tuning are not required due to its
high level of abstraction, which makes it easier to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'
nan_cache = {}


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	# Changing 'male' to 1, 'female' to 0
	data.loc[data.Sex == 'male', 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0
	# Changing 'S ' to 0, 'C' to 1, 'Q' to 2
	data.loc[data.Embarked == 'S', 'Embarked'] = 0
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2

	data.drop('PassengerId', axis=1, inplace=True)
	data.drop('Name', axis=1, inplace=True)
	data.drop('Ticket', axis=1, inplace=True)
	data.drop('Cabin', axis=1, inplace=True)
	if mode == 'Train':
		data = data.dropna()
		labels = data.pop('Survived')
		nan_cache['Age'] = round(data['Age'].mean(), 3)
		nan_cache['Fare'] = round(data['Fare'].mean(), 3)
		return data, labels

	elif mode == 'Test':
		key_need = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
		data = data[key_need]
		data['Age'].fillna(nan_cache['Age'], inplace=True)
		data['Fare'].fillna(nan_cache['Fare'], inplace=True)
	return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	if feature == 'Sex':
		data['Sex_0'] = 0
		data.loc[data.Sex == 0, 'Sex_0'] = 1
		# One hot encoding for a new category Female
		data['Sex_1'] = 0
		data.loc[data.Sex == 1, 'Sex_1'] = 1
		# No need Sex anymore!
		data.pop('Sex')
	# One hot encoding for a new category FirstClass
	elif feature == 'Pclass':
		data['Pclass_0'] = 0
		data.loc[data.Pclass == 1, 'Pclass_0'] = 1
		# One hot encoding for a new category SecondClass
		data['Pclass_1'] = 0
		data.loc[data.Pclass == 2, 'Pclass_1'] = 1
		# One hot encoding for a new category ThirdClass
		data['Pclass_2'] = 0
		data.loc[data.Pclass == 3, 'Pclass_2'] = 1
		# No need Pclass anymore!
		data.pop('Pclass')
	# One hot encoding for a new category Embarked
	elif feature == 'Embarked':
		data['Embarked_0'] = 0
		data.loc[data.Embarked == 0, 'Embarked_0'] = 1
		# One hot encoding for a new category SecondClass
		data['Embarked_1'] = 0
		data.loc[data.Embarked == 1, 'Embarked_1'] = 1
		# One hot encoding for a new category ThirdClass
		data['Embarked_2'] = 0
		data.loc[data.Embarked == 2, 'Embarked_2'] = 1
		data.pop('Embarked')
	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	standardizer = preprocessing.StandardScaler()
	data = standardizer.fit_transform(data)
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy on degree1;
	~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimal places)
	TODO: real accuracy on degree1 -> 0.80196629
	TODO: real accuracy on degree2 -> 0.83848315
	TODO: real accuracy on degree3 -> 0.87219101
	"""
	train_data, Y = data_preprocess(TRAIN_FILE, mode='Train')
	print(train_data)
	test_data = data_preprocess(TEST_FILE, mode='Test')
	train_data = one_hot_encoding(train_data, 'Sex')
	train_data = one_hot_encoding(train_data, 'Pclass')
	train_data = one_hot_encoding(train_data, 'Embarked')
	# # Degree 1 Polynomial Model #
	# #############################
	standardizer = preprocessing.StandardScaler()
	X_sta = standardizer.fit_transform(train_data)
	h = linear_model.LogisticRegression()
	classifier = h.fit(X_sta, Y)
	train_acc_d1 = classifier.score(X_sta, Y)
	print('real accuracy on degree1', round(train_acc_d1, 8))

	# # Degree 2 Polynomial Model #
	# #############################
	poly2_phi = preprocessing.PolynomialFeatures(degree=2)
	X_poly2 = poly2_phi.fit_transform(X_sta)
	classifier2 = h.fit(X_poly2, Y)
	train_acc_d2 = classifier2.score(X_poly2, Y)
	print('real accuracy on degree2', round(train_acc_d2, 8))

	# # Degree 3 Polynomial Model #
	# #############################
	poly3_phi = preprocessing.PolynomialFeatures(degree=3)
	X_poly3 = poly3_phi.fit_transform(X_sta)
	classifier3 = h.fit(X_poly3, Y)
	train_acc_d3 = classifier3.score(X_poly3, Y)
	print('real accuracy on degree3', round(train_acc_d3, 8))


if __name__ == '__main__':
	main()
