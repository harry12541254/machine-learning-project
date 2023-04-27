"""
File: titanic_level1.py
Name: 
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle website. This model is the most flexible among all
levels. You should do hyper-parameter tuning to find the best model.
"""

import math
from collections import defaultdict

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'

def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating the mode we are using
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	filename = TRAIN_FILE if mode == 'Train'else TEST_FILE
	data = defaultdict(list)
	with open(filename, 'r') as f:
		first_line = True
		for line in f:
			line = line.strip()
			if first_line:
				first_line = False
			else:
				lst = line.split(',')
				# The line will be ignore if Age value or Embarked value is missing data in Train mode
				if not lst[6] or not lst[len(lst) - 1]:
					continue
				data['PassengerId'].append(int(lst[0]))
				if mode == 'Train':
					data['Survived'].append(int(lst[1]))
					start = 2
				else:
					start = 1
				for i in range(len(lst)):
					if i == start:
						data['Pclass'].append(int(lst[i]))
					elif i == start + 1:
						data['Name'].append(lst[i])
					elif i == start + 3:
						data['Sex'].append(1) if lst[i] == 'male' else data['Sex'].append(0)
					elif i == start + 4:
						data['Age'].append(float(lst[i])) if lst[i] != '' else data['Age'].append(29.642)
					elif i == start + 5:
						data['SibSp'].append(int(lst[i]))
					elif i == start + 6:
						data['Parch'].append(int(lst[i]))
					elif i == start + 7:
						data['Ticket'].append(lst[i])
					elif i == start + 8:
						data['Fare'].append(float(lst[i])) if lst[i] != '' else data['Fare'].append(34.567)
					elif i == start + 9:
						data['Cabin'].append(lst[i])
					else:
						if i == start + 10:
							if lst[i] == 'S':
								data['Embarked'].append(0)
							elif lst[i] == 'C':
								data['Embarked'].append(1)
							else:
								data['Embarked'].append(2)
		if mode == 'Train' or mode == 'Test':
			data.pop('PassengerId')
			data.pop('Name')
			data.pop('Ticket')
			data.pop('Cabin')
		return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	copy_data = data.copy()
	data = defaultdict(list, data)
	for key, value in data.items():
		if key == feature:
			for ch in data['Sex']:
				if ch == 1:
					copy_data['Sex_1'].append(int(1))
					copy_data['Sex_0'].append(int(0))
				else:
					copy_data['Sex_1'].append(int(0))
					copy_data['Sex_0'].append(int(1))
			for ch in data['Pclass']:
				if ch == 1:
					copy_data['Pclass_0'].append(int(1))
					copy_data['Pclass_1'].append(int(0))
					copy_data['Pclass_2'].append(int(0))
				elif ch == 2:
					copy_data['Pclass_0'].append(int(0))
					copy_data['Pclass_1'].append(int(1))
					copy_data['Pclass_2'].append(int(0))
				else:
					copy_data['Pclass_0'].append(int(0))
					copy_data['Pclass_1'].append(int(0))
					copy_data['Pclass_2'].append(int(1))
			for ch in data['Embarked']:
				if ch == 0:
					copy_data['Embarked_0'].append(int(1))
					copy_data['Embarked_1'].append(int(0))
					copy_data['Embarked_2'].append(int(0))
				elif ch == 1:
					copy_data['Embarked_0'].append(int(0))
					copy_data['Embarked_1'].append(int(1))
					copy_data['Embarked_2'].append(int(0))
				else:
					copy_data['Embarked_0'].append(int(0))
					copy_data['Embarked_1'].append(int(0))
					copy_data['Embarked_2'].append(int(1))
			copy_data.pop('Sex')
			copy_data.pop('Pclass')
			copy_data.pop('Embarked')
	return copy_data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	for key, value in data.items():
		max_x = max(value)
		min_x = min(value)
		# if max_x > 1:
		for i in range(len(value)):
			data[key][i] = (value[i] - min_x) / (max_x - min_x)
	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature -> weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0
	# Step 2 : Start training
	for epoch in range(num_epochs):
		for num in range(len(labels)):
			if degree == 1:
				# Step 3: Feature extract
				feature_vector_d1 = featureExtractor(inputs, num, degree)
				# Step 4: Weights updating
				k = dotProduct(feature_vector_d1, weights)
				h = 1/(1+math.exp(-k))
				increment(weights, (- alpha * (h - labels[num])), feature_vector_d1)
			else:
				# Step 3: Feature extract
				feature_vector_d2 = featureExtractor(inputs, num, degree)
				# Step 4: Weights updating
				k = dotProduct(feature_vector_d2, weights)
				h = 1 / (1 + math.exp(-k))
				increment(weights, (- alpha * (h - labels[num])), feature_vector_d2)
	return weights


def featureExtractor(inputs, num, degree):
	if degree == 1:
		phi_vector_d1 = defaultdict(float)
		for key, value in inputs.items():
			phi_vector_d1[key] = value[num]
		return phi_vector_d1
	elif degree == 2:
		phi_vector_d2 = defaultdict(float)
		keys = list(inputs.keys())
		values = list(inputs.values())
		for key, value in inputs.items():
			phi_vector_d2[key] = value[num]
			values = list(phi_vector_d2.values())
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				phi_vector_d2[keys[i] + keys[j]] = values[i] * values[j]
		# Will return {''Age':0.2,'SibSp':0,'Parch':1,...'AgeAge':0.068, 'AgeSibSp':0, 'AgeFare':0.34}
		return phi_vector_d2


def dotProduct(d1, d2):
	"""
    @param dict d1: a feature vector. Key is a feature (string); value is its weight (float).
    @param dict d2: a feature vector. Key is a feature (string); value is its weight (float)
    @return float: the dot product between d1 and d2
    """
	if len(d1) < len(d2):
		return dotProduct(d2, d1)
	else:
		# BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
		return sum(d1.get(key, 0)*value for key, value in d2.items())
		# END_YOUR_CODE


def increment(d1, scale, d2):
	"""
	Implements d1 += scale * d2 for sparse vectors.
	@param dict d1: the feature vector which is mutated.
	@param scale: float, scale value of d2 to add onto the corresponding value of d1
	@param dict d2: a feature vector.
	"""
	# BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
	for key, value in d2.items():
		d1[key] = d1.get(key, 0) + scale * value
	# END_YOUR_CODE