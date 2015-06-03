import re
import numpy as np
import math

def load(fn):
	ds = []
	f = open(fn, "r")
	for line in f:
		tokens = line.split()
		features = []
		for i in range(len(tokens) - 1):
			features.append(int(tokens[i]))
		data = (features, int(tokens[-1]))
		ds.append(data)
	return ds

def load_dictionary(fn):
	ds = []
	f = open(fn, "r")
	for line in f:
		tokens = line.split()
		ds.append(tokens[0])
	return ds

def sum_weight(weights):
	sum = float(0)
	for weight in weights:
		sum = sum + weight
	return sum

def weak_learner(i, dataset, weights):
	error = float(0)
	length = len(dataset)
	for j in range(0, length):
		(features, label) = dataset[j]
		if (features[i] == 1 and label == -1) or (features[i] == 0 and label == 1) :
			# for H_i+ only correct if the feature == label i.e if the word
			# is in the feature and the label is positive or the word
			# is not there and the label is -1
			error = error + weights[j]

	if error > float(.5):
		return (-1, 1- error)
	else:
		return (1, error)
 
def init_weights(dataset):
 	initial = float(1)/len(dataset)
 	w = [initial] * len(dataset)
 	return w
 
def calc_alpha(error):
 	return float(.5) * np.log((1-error)/error)

def calc_next_d(alpha, classifier, word, dataset, prev_weights):
	d = []
	for index in range(0, len(dataset)):
		(features, label) = dataset[index]
		sign = 0
		if (features[word] == 1 and classifier == 1 and label == 1) or (features[word] == 0 and classifier == 1 and label == -1) or (features[word] == 1 and classifier == -1 and label == -1) or (features[word] == 0 and classifier == -1 and label == 1):
			sign = 1
		else:
			sign = -1
		d_val = prev_weights[index] * math.exp(-1 * alpha * sign)
		d.append(d_val)
	return d
	
def test(final_classifiers, dataset):
	errors = 0
	num_samples = float(len(dataset))
	for (features, label) in dataset:
		count = 0
		for (alpha, classifier_type, word) in final_classifiers:
			prediction = 0 
			if (classifier_type == -1 and features[word] == 0) or (classifier_type == 1 and features[word] == 1):
				prediction = 1
			else: 
				prediction = -1
			count = count + alpha * prediction
		if (count < 0 and label == 1) or (count > 0 and label == -1):
			errors = errors + 1
	return errors/num_samples

def main():
	training_set = load("hw6train.txt")	
	test_set = load("hw6test.txt")
	dictionary = load_dictionary("hw6dictionary.txt")
	weights = init_weights(training_set)
	final = []

	for rounds in range(0,20):
		(best_classifier, best_error, word) = (100, float(2), "default")
		for i in range(0, len(training_set[0][0])):
			(classifier, error) = weak_learner(i, training_set, weights)
			if (error < best_error):
				(best_classifier, best_error, word) = (classifier, error, i)
		alpha = calc_alpha(best_error)	
		d = calc_next_d(alpha, best_classifier, word, training_set, weights)
		z = sum_weight(d)
		d = [d_val / z for d_val in d]
		final.append((alpha, best_classifier, word))
		weights = d
		if rounds in [2, 6, 9, 14, 19]:
			print "Round " + str(rounds + 1) + " :"
			print "Training Error: " + str(test(final, training_set))
			print "Testing Error: " + str(test(final, test_set))
	


if __name__ == '__main__':
	main()