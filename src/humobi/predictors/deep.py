import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def gen(x_data, n_splits):
	"""
	Generates data for learning.

	Args:
		x_data: DataFrame of data for learning
		n_splits: the number of splits

	Returns:
		training and testing datasets
	"""
	for train_index, test_index in KFold(n_splits).split(x_data):
		x_train, x_test = x_data[train_index], x_data[test_index]
		yield x_train, x_test


def split_input_target(chunk):
	"""
	Splits data into chunks of inputs and target values.

	Args:
		chunk: chunk to split

	Returns:
		split data
	"""
	# for the example: hello
	input_text = chunk[:-1]  # hell
	target_text = chunk[1:]  # ello
	return input_text, target_text  # hell, ello


class GRUModel2(tf.keras.Model):
	"""
	GRU Model (2 GRU layers + embedding + dropout layer)

	Args:
		vocab_size: Vocabulary size - the number of unique places in sequence
		embedding_dim: The size of embedding layer
		rnn_units: The number of rnn units on a hidden layer
		dropout: Dropout size
		batch_size: Size of a single batch
	"""

	def __init__(self, vocab_size, embedding_dim, rnn_units, dropout, batch_size, window_size):
		"""
		Layer structure.
		"""
		super().__init__(self)
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=(batch_size,vocab_size))
		self.gru = tf.keras.layers.GRU(rnn_units,
		                               return_sequences=True,
		                               return_state=True, batch_input_shape=(batch_size, window_size, embedding_dim))
		self.gru2 = tf.keras.layers.GRU(rnn_units,
		                                return_sequences=True,
		                                return_state=True, batch_input_shape=(batch_size, window_size, embedding_dim))
		self.drop = tf.keras.layers.Dropout(dropout)
		self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

	def call(self, inputs, states=None, return_state=False, training=False):
		"""
		Constructs the network for training.

		Args:
			inputs: Input vectors
			states: Initial states. If none, no initial state is set.
			return_state (default = False): Boolean. Whether to return the last state in addition to the output.
			training (default = False): Boolean. Indicating whether the layer should behave in training mode or in inference mode

		Returns:
			The GRU network
		"""
		x = inputs
		x = self.embedding(x)
		if states is None:
			states = self.gru.get_initial_state(x)
		x, states = self.gru(x, initial_state=states, training=training)
		x, states = self.gru2(x, initial_state=states, training=training)
		x = self.drop(x, training=training)
		x = self.dense(x)

		if return_state:
			return x, states
		else:
			return x


class GRUModel(tf.keras.Model):
	"""
		GRU Model (1 GRU layer + embedding + dropout layer)

	Args:
		vocab_size: Vocabulary size - the number of unique places in sequence
		embedding_dim: The size of embedding layer
		rnn_units: The number of rnn units on a hidden layer
		dropout: Dropout size
		batch_size: Size of a single batch
	"""

	def __init__(self, vocab_size, embedding_dim, rnn_units, dropout, batch_size):
		super().__init__(self)
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(rnn_units,
		                               return_sequences=True,
		                               return_state=True, stateful=True)
		self.drop = tf.keras.layers.Dropout(dropout)
		self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

	def call(self, inputs, states=None, return_state=False, training=False):
		"""
		Constructs the network for training.

		Args:
			inputs: Input vectors
			states: Initial states. If none, no initial state is set.
			return_state (default = False): Boolean. Whether to return the last state in addition to the output.
			training (default = False): Boolean. Indicating whether the layer should behave in training mode or in inference mode

		Returns:
			The GRU network
		"""
		x = inputs
		x = self.embedding(x)
		if states is None:
			states = self.gru.get_initial_state(x)
		x, states = self.gru(x, initial_state=states, training=training)
		x = self.drop(x, training=training)
		x = self.dense(x, training=training)

		if return_state:
			return x, states
		else:
			return x


def loss(labels, logits):
	"""
	The loss function - categorical crossentropy.

	Args:
		labels: ground truth values
		logits: predictions encoded as probability distribution

	Returns:
		loss values
	"""
	return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


class DeepPred():
	"""
	Deep-learning based sequence prediction model

	Args:
		model: Name of the model to use. Available: GRU (single layer), GRU2 (GRU double layer), LSTM (single layer)
		data: Data for prediction - whole dataset, including test. Do not split it.
		test_size: Test ratio for data split
		folds: The number of folds in cross-validation of the model
		window_size: Window size, also referenced as the size of a lookback. The number of previous symbols considered
		during prediction.
		batch_size: The size of a batch for a single network
		embedding_dim: The size of an embedding layer (the number of nodes)
		rnn_units:  The number of RNN layer's units
	"""

	def __init__(self, model, data, test_size=.2, folds=5, window_size=2, batch_size=1, embedding_dim=1024,
	             rnn_units=1024):
		"""
		Class initialisation. Runs data preparation - slicing into training and test sets.
		"""
		self.model = model
		self.data = data
		self.split_ratio = test_size
		self.folds = folds
		self.window_size = window_size
		self.batch_size = batch_size
		self.embedding_dim = embedding_dim
		self.rnn_units = rnn_units
		self.scores = {}
		self.predictions = {}
		self._prepare()

	def _prepare(self):
		"""
		Prepares data for prediction - splits data into train and test sets.
		"""
		data_len = self.data.groupby(level=0).apply(lambda x: len(x))
		cut_index = (1-round(self.split_ratio * data_len)).astype(int)
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
		dataset_train = [t.map(split_input_target) for t in sequences_train]  # and convert them using the windowing algorithm
		dataset_val = [v.map(split_input_target) for v in sequences_val]
		dataset_test = sequences_test.map(split_input_target)  # data test is split but not used - just for debugging
		data_tr = [t.shuffle(1000).batch(self.batch_size, drop_remainder=True) for t in
		           dataset_train]  # shuffle the data for training and batch
		data_val = [v.shuffle(1000).batch(self.batch_size, drop_remainder=True) for v in
		            dataset_val]  # shuffle the data for testing and batch
		fold = 1  # the counter of folds
		callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)  # initialise the EarlyStopping mechanism
		# Initialiase selected models - the dropout rate is fixed
		if self.model == "GRU":
			model = GRUModel(self.s_regions, self.embedding_dim, self.rnn_units, 0.2, self.batch_size)
		elif self.model == "GRU2":
			model = GRUModel2(self.s_regions, self.embedding_dim, self.rnn_units, 0.2, self.batch_size, self.window_size)
		model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])  # compile the model
		for x, v in zip(data_tr, data_val):  # the CV training process
			print("FOLD:{}".format(fold))
			self.history = model.fit(x, epochs=30, validation_data=v, callbacks=[callback], batch_size=self.batch_size)
			fold += 1
		model.reset_states()
		return model

	def learn_predict(self):
		"""
		Trains the network and makes prediction on the test set. The prediction is given a fixed temperature = 0.1, you
		can change it here. The higher the more random are predictions. However, small temperature can cause network
		to stuck in an infinite loop.
		"""
		result_dic = {}
		predictions_dic = {}
		train = self.data[0].groupby(level=0)
		test = self.data[1].groupby(level=0)
		for tr, ts in zip(train, test):
			uid = tr[0]
			tr_data = tr[1].values.astype(int)
			ts_data = ts[1].values.astype(int)
			user_model = self._user_learn(tr_data, ts_data)
			stabs = []
			temperature = .1 # TEMPERATURE
			for stab in range(1): # TODO: add as parameter
				forecast = []
				for x in range(len(ts_data) - self.window_size):
					y = tf.expand_dims(ts_data[x:x + self.window_size], 0)
					predictions = user_model.predict(y, batch_size=self.batch_size)
					predictions = np.squeeze(predictions, 0)
					predictions = predictions / temperature
					predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
					forecast.append(predicted_id)
				stabs.append(sum(forecast == ts_data[self.window_size:]) / len(forecast))
			predictions_dic[uid] = (forecast,ts_data[self.window_size:])
			print(np.mean(stabs))
			result_dic[uid] = np.mean(stabs)
		self.scores = pd.DataFrame.from_dict(result_dic, orient='index')
		self.predictions = pd.concat({k: pd.concat([pd.Series(v[0]),pd.Series(v[1])],axis=1) for k,v in predictions_dic.items()}).droplevel(1)
