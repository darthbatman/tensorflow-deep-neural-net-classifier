from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import random

import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

tf.logging.set_verbosity(tf.logging.ERROR)

training_set_raw = []

for i in range(0, 120):
	wheels = random.randint(2, 4)
	length = random.randint(100, 1000)
	height = random.randint(100, 1000)
	width = random.randint(100, 1000)
	vehicle_type = -1
	if wheels == 2:
		vehicle_type = 1
	elif wheels == 3:
		vehicle_type = 2
	elif wheels == 4:
		vehicle_type = 0
	training_set_raw.append([wheels, length, height, width, vehicle_type])

test_set_raw = [[3, 100, 200, 100, 2], \
	[3, 100, 105, 100, 2], \
	[2, 800, 303, 300, 1], \
	[2, 900, 706, 600, 1], \
	[2, 100, 202, 800, 1], \
	[2, 700, 100, 900, 1], \
	[4, 500, 700, 800, 0], \
	[4, 400, 300, 500, 0], \
	[4, 900, 700, 500, 0], \
	[4, 600, 400, 400, 0]]

f = open('vehicles_training.csv', 'w')
f.write(str(len(training_set_raw)) + ',' + str(len(training_set_raw[0]) - 1) + ',car,motorcyle,trike\n')
for i in range(0, len(training_set_raw)):
	for j in range(0, len(training_set_raw[i])):
		f.write(str(training_set_raw[i][j]))
		if (j != (len(training_set_raw[i]) - 1)):
			f.write(',')
		else:
			f.write('\n')
f.close()

f = open('vehicles_test.csv', 'w')
f.write(str(len(test_set_raw)) + ',' + str(len(test_set_raw[0]) - 1) + ',car,motorcyle,trike\n')
for i in range(0, len(test_set_raw)):
	for j in range(0, len(test_set_raw[i])):
		f.write(str(test_set_raw[i][j]))
		if (j != (len(test_set_raw[i]) - 1)):
			f.write(',')
		else:
			f.write('\n')
f.close()

file = open("vehicles_training.csv", "r") 
print(file.read())

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename="vehicles_training.csv", target_dtype=np.int, features_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename="vehicles_test.csv", target_dtype=np.int, features_dtype=np.int)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10],n_classes=3,model_dir="/tmp/vehicle_model")

classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

def new_samples():
  return np.array(
    [[4, 100, 200, 300],
     [2, 999, 121, 232]], dtype=np.int)

predictions = list(classifier.predict(input_fn=new_samples))

print("New Samples, Class Predictions:    {}\n".format(predictions))