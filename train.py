#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense

from graph_sequence import *

def generate_model():
	model = Sequential([
		Dense(6, 
			input_shape=(12,),
			activation='tanh'),
		Dense(6, activation='tanh'),
		Dense(1, activation='sigmoid'),
	])

	return model

def train(args):

	model = generate_model()

	model.compile(loss='mean_squared_error',
				optimizer='adam',
				metrics=['accuracy'])

	seq_train = GraphSequence(args)
	model.fit_generator(seq_train, epochs=10)

	seq_test = GraphSequence(args, test=True)
	result = model.evaluate_generator(seq_test)

	print(f"Accuracy: {round(result[1]*100)}%")


if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--database', type=str, default="hosted")
	args = parser.parse_args()

	train(args)


