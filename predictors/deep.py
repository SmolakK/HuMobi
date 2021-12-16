import tensorflow as tf
import numpy as np
import pandas as pd
import os
from structures.trajectory import TrajectoriesFrame
from tools.user_statistics import count_records
from misc.utils import to_labels
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import kerastuner as kt

sys.path.append("..")


def gen(x_data, n_splits):
	"""
	Generates data for learning
	:param x_data: DataFrame of data for learning
	:param n_splits: the number of splits
	:return: training and testing datasets
	"""
	for train_index, test_index in KFold(n_splits).split(x_data):
		x_train, x_test = x_data[train_index], x_data[test_index]
		yield x_train, x_test


def split_input_target(chunk):
	"""
	Splits data into chunks of inputs and target values
	:param chunk: chunk to split
	:return: split data
	"""
	# for the example: hello
	input_text = chunk[:-1]  # hell
	target_text = chunk[1:]  # ello
	return input_text, target_text  # hell, ello


class LSTModel2(tf.keras.Model):
	"""
	LSTM Model (2 LSTM layers + embedding + dropout layer)
	"""

	def __init__(self, vocab_size, embedding_dim, rnn_units, dropout):
		super().__init__(self)
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.lstm = tf.keras.layers.LSTM(rnn_units,
		                                 return_sequences=True, stateful=True)
		self.lstm = tf.keras.layers.LSTM(rnn_units,
		                                 return_sequences=True, stateful=True)
		self.drop = tf.keras.layers.Dropout(dropout)
		self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

	def call(self, inputs, states=None, return_state=False, training=False):
		x = inputs
		x = self.embedding(x)
		if states is None:
			states = self.lstm.get_initial_state(x)
		x = self.lstm(x, initial_state=states, training=training)
		x = self.lstm(x, initial_state=states, training=training)
		x = self.dense(x)

		if return_state:
			return x, states
		else:
			return x


class GRUModel2(tf.keras.Model):
	"""
	GRU Model (2 GRU layers + embedding + dropout layer)
	"""

	def __init__(self, vocab_size, embedding_dim, rnn_units, dropout):
		super().__init__(self)
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(rnn_units,
		                               return_sequences=True,
		                               return_state=True, stateful=True)
		self.gru2 = tf.keras.layers.GRU(rnn_units,
		                                return_sequences=True,
		                                return_state=True, stateful=True)
		self.drop = tf.keras.layers.Dropout(dropout)
		self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

	def call(self, inputs, states=None, return_state=False, training=False):
		x = inputs
		x = self.embedding(x)
		if states is None:
			states = self.gru.get_initial_state(x)
		x, states = self.gru(x, initial_state=states, training=training)
		x, states = self.gru2(x, initial_state=states, training=training)
		x = self.dense(x)

		if return_state:
			return x, states
		else:
			return x


class GRUModel(tf.keras.Model):
	"""
		GRU Model (1 GRU layer + embedding + dropout layer)
	"""

	def __init__(self, vocab_size, embedding_dim, rnn_units, dropout):
		super().__init__(self)
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(rnn_units,
		                               return_sequences=True,
		                               return_state=True, stateful=True)
		self.drop = tf.keras.layers.Dropout(dropout)
		self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

	def call(self, inputs, states=None, return_state=False, training=False):
		x = inputs
		x = self.embedding(x)
		if states is None:
			states = self.gru.get_initial_state(x)
		x, states = self.gru(x, initial_state=states, training=training)
		x = self.dense(x)

		if return_state:
			return x, states
		else:
			return x


def loss(labels, logits):
	"""
	The loss function - categorical crossentropy
	:param labels: ground truth values
	:param logits: predictions encoded as probability distribution
	:return: loss values
	"""
	return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


class DeepPred():
	"""
	Deep-learning based sequence prediction model

	:param model: Name of the model to use. Available: GRU (single layer), GRU2 (GRU double layer), LSTM (single layer)
	:param data: Data for prediction - whole dataset, including test
	:param test_ratio: Test ratio for data split
	:param folds: The number of folds in cross-validation of the model
	:param window_size: Window size, also referenced as the size of a lookback. The number of previous symbols considered
	during prediction.
	:param batch_size: The size of a batch for a single network
	:param embedding_dim: The size of an embedding layer (the number of nodes)
	:param rnn_units:  The number of RNN layer's units
	"""

	def __init__(self, model, data, test_ratio=.8, folds=5, window_size=2, batch_size=1, embedding_dim=1024,
	             rnn_units=1024):
		"""
		Class initialisation. Runs data preparation - slicing into training and test sets.
		"""
		self.model = model
		self.data = data
		self.test_ratio = test_ratio
		self.folds = folds
		self.window_size = window_size
		self.batch_size = batch_size
		self.embedding_dim = embedding_dim
		self.rnn_units = rnn_units
		self.scores = {}
		self.prepare()

	def prepare(self):
		"""
		Prepares data for prediction - splits data into train and test sets.
		"""
		data_len = self.data.groupby(level=0).apply(lambda x: len(x))
		cut_index = round(self.test_ratio * data_len).astype(int)
		data_train = self.data.groupby(level=0).apply(
			lambda x: x.labels.iloc[:cut_index.loc[x.index.get_level_values(0)[0]]])
		data_test = self.data.groupby(level=0).apply(
			lambda x: x.labels.iloc[cut_index.loc[x.index.get_level_values(0)[0]] - self.window_size:])
		self.data = (data_train, data_test)

	def _user_learn(self, tr_data, ts_data):
		ts_dataset = tf.data.Dataset.from_tensor_slices(ts_data)  # transform test set into tensor
		if self.folds != 1:  # if there is more than single fold
			train_folds = [(tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y)) for x, y in
			               gen(tr_data, self.folds)]  # then split data into the list folds
		else:  # if there is a single fold
			train_folds = [[(tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y)) for x, y in
			                gen(tr_data, 2)][0]]  # make it a one set of data
		self.s_regions = np.hstack([tr_data, ts_data]).max() + 1  # read the number of unique symbols from the dataset
		sequences_train = [dataset_train[0].batch(self.window_size + 1, drop_remainder=True) for dataset_train in
		                   train_folds]  # prepare the batches of sequences for training
		sequences_val = [dataset_train[1].batch(self.window_size + 1, drop_remainder=True) for dataset_train in
		                 train_folds]  # prepare the batches of sequences for validation
		sequences_test = ts_dataset.batch(self.window_size + 1, drop_remainder=True)  # prepare the batches of
		# sequences for tests
		dataset_train = [t.map(split_input_target) for t in
		                 sequences_train]  # and convert them using the windowing algorithm
		dataset_val = [v.map(split_input_target) for v in sequences_val]
		dataset_test = sequences_test.map(split_input_target)  # data test is split but not used - just for debugging
		data_tr = [t.shuffle(1000).batch(self.batch_size, drop_remainder=True) for t in dataset_train]  # shuffle the data for training and batch
		data_val = [v.shuffle(1000).batch(self.batch_size, drop_remainder=True) for v in dataset_val]  # shuffle the data for testing and batch
		fold = 1  # the counter of folds
		callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)  # initialise the EarlyStopping mechanism
		# Initialiase selected models - the dropout rate is fixed
		if self.model == "GRU":
			model = GRUModel(self.s_regions, self.embedding_dim, self.rnn_units, 0.2)
		elif self.model == "LSTM":
			model = LSTModel2(self.s_regions, self.embedding_dim, self.rnn_units, 0.2)
		elif self.model == "GRU2":
			model = GRUModel2(self.s_regions, self.embedding_dim, self.rnn_units, 0.2)
		model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])  # compile the model
		for x, v in zip(data_tr, data_val):  # the CV training process
			print("FOLD:{}".format(fold))
			history = model.fit(x, epochs=30, validation_data=v, callbacks=[callback])
			fold += 1
		return model

	def learn_predict(self):
		"""
		Trains the network and makes prediction on the test set. The prediction is given a fixed temperature = 0.1, you
		can change it here. The higher the more random are predictions. However, small temperature can cause network
		to stuck in an infinite loop.
		"""
		result_dic = {}
		train = self.data[0].groupby(level=0)
		test = self.data[1].groupby(level=0)
		for tr, ts in zip(train, test):
			uid = tr[0]
			tr_data = tr[1].values.astype(int)
			ts_data = ts[1].values.astype(int)
			user_model = self._user_learn(tr_data, ts_data)
			stabs = []
			temperature = .1
			for stab in range(10):
				forecast = []
				for x in range(len(ts_data) - self.window_size):
					y = tf.expand_dims(ts_data[x:x + self.window_size], 0)
					predictions = user_model(y)
					predictions = np.squeeze(predictions, 0)
					predictions = predictions / temperature
					predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
					forecast.append(predicted_id)
				stabs.append(sum(forecast == ts_data[self.window_size:]) / len(forecast))
			result_dic[uid] = np.mean(stabs)
		self.scores = pd.DataFrame.from_dict(result_dic, orient='index')
