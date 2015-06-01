import re

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

def main():
	training_set = load("hw6train.txt")	
	test_set = load("hw6test.txt")
if __name__ == '__main__':
	main()