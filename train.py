#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense

from graph_sequence import *

def generate_model():
	model = Sequential([
		Dense(4, 
			input_shape=(4,),
			activation='tanh'),
		Dense(4, activation='tanh'),
		Dense(1, activation='sigmoid'),
	])

	return model

def train():

	seq_train = GraphSequence()
	seq_test = GraphSequence(test=True)

	model = generate_model()

	model.compile(loss='mean_squared_error',
				optimizer='adam',
				metrics=['accuracy'])

	model.fit_generator(seq_train, epochs=2)
	result = model.evaluate_generator(seq_test)

	print(f"Accuracy: {round(result[1]*100)}")


if __name__ == '__main__':
	train()


