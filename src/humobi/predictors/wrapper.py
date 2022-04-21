import numpy as np
import pandas as pd
from humobi.misc.utils import to_labels
from tqdm import tqdm
tqdm.pandas()
from humobi.predictors.markov import MarkovChain
from humobi.predictors.sparse import Sparse
from sklearn.model_selection import TimeSeriesSplit
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import concurrent.futures as cf


def iterate_random_values(S, n_check):
	"""
	Takes a random combination of items of a given size.

	Args:
		S: The dictionary of items
		n_check: Size of combination

	Returns:
		A random combination
	"""
	keys, values = zip(*S.items())
	combs = [dict(zip(keys, row)) for row in itertools.product(*values)]
	combs = np.random.choice(combs, n_check)
	return combs


class Splitter():
	"""
	This is a Splitter class responsible for data preparation for shallow machine learning-based models from sklearn library.

	Args:
		trajectories_frame: TrajectoriesFrame class object
		split_ratio: The ratio of training data
		horizon: Window size - determines how many previous symbols are considered by the model when predicting
		n_splits: The number of splits for cross-validation process
	"""

	def __init__(self, trajectories_frame, split_ratio, horizon = 1, n_splits = 1):
		"""
		Class initialisation. Calls data splitting routine, hence after initialisation data is already prepared.
		"""
		self._data = trajectories_frame
		self._train_ratio = 1 - split_ratio
		if horizon < 1:
			raise ValueError("Horizon value has to be a positive integer")
		self._horizon = horizon
		self._n_splits = n_splits
		self.cv_data = []
		self._stride_data()

	@property
	def data(self):
		return self._data

	@property
	def test_ratio(self):
		return self._train_ratio

	@property
	def horizon(self):
		return self._horizon

	def _test_split(self, X, Y):
		"""
		Splits test data into training and testing sets. Uses windowed data.

		Args:
			X: Windowed features
			Y: Windowed targets

		Returns:
			Training and testing sets of data
		"""
		train_frame_X = X.groupby(level=0).progress_apply(lambda x: x.iloc[:round(len(x) * self.test_ratio)]).droplevel(
			1)
		train_frame_Y = Y.groupby(level=0).progress_apply(lambda x: x.iloc[:round(len(x) * self.test_ratio)]).droplevel(
			1)
		test_frame_X = X.groupby(level=0).progress_apply(lambda x: x.iloc[round(len(x) * self.test_ratio):]).droplevel(
			1)
		test_frame_Y = Y.groupby(level=0).progress_apply(lambda x: x.iloc[round(len(x) * self.test_ratio):]).droplevel(
			1)
		return train_frame_X, train_frame_Y, test_frame_X, test_frame_Y

	def _cv_split(self, frame_X, frame_Y, n_splits=5):
		"""
		Splits training data into the training and validation sets using cross-validation approach.

		Args:
			frame_X: Training features
			frame_Y: Training targets
			n_splits: The number of splits to be applied
		"""
		for n in range(1, n_splits + 1):
			train_set_X = frame_X.groupby(level=0).apply(
				lambda x: x.iloc[:round(x.shape[0] * (n / (n_splits + 1)))]).droplevel(0)
			train_set_Y = frame_Y.groupby(level=0).apply(
				lambda x: x.iloc[:round(x.shape[0] * (n / (n_splits + 1)))]).droplevel(0)
			val_set_X = frame_X.groupby(level=0).apply(lambda x: x.iloc[round(x.shape[0] * (n / (n_splits + 1))):round(
				x.shape[0] * ((n + 1) / (n_splits + 1)))]).droplevel(0)
			val_set_Y = frame_Y.groupby(level=0).apply(lambda x: x.iloc[round(
				x.shape[0] * (n / (n_splits + 1))):round(
				x.shape[0] * ((n + 1) / (n_splits + 1)))]).droplevel(0)
			self.cv_data.append((train_set_X, train_set_Y, val_set_X, val_set_Y))

	def _stride_data_single(self, frame):
		"""
		Uses windowing algorithm to prepare time-series data from sequences to prediction. Takes labels and horizon size
		to create chunks of data.

		Args:
			frame: TrajectoriesFrame of single user

		Returns:
			Chunks of data in a DataFrame - features and targets
		"""
		to_concat = []
		for uid, traj in frame.groupby(level=0):
			traj = traj.reset_index()
			for x in range(1, self.horizon + 1):
				traj['labels{}'.format(x)] = traj['labels'].shift(-x)
			traj['datetime'] = traj['datetime'].shift(-self.horizon)
			traj = traj.set_index(['user_id', 'datetime'])
			to_concat.append(traj[:-self.horizon])
		frame_ready = pd.concat(to_concat)
		frame_X = frame_ready.iloc[:, :self.horizon]
		frame_Y = frame_ready.iloc[:, -1]
		return frame_X, frame_Y

	def _stride_data(self):
		"""
		A wrapper function for data windowing algorithm. Calls windowing algorithm for every unique user in the dataset.
		After it splits data into three datasets - test, train and validation.
		"""
		strided_X, strided_Y = self._stride_data_single(self.data['labels'])
		train_frame_X, train_frame_Y, self.test_frame_X, self.test_frame_Y = self._test_split(strided_X, strided_Y)
		self._cv_split(train_frame_X, train_frame_Y, n_splits=self._n_splits)


class SKLearnPred():
	"""
	This is a wrapper for classification function from sklearn library which can be used for prediction here.

	Args:
		algorithm: An algorithm from sklearn library
		training_data: The training set from Splitter class
		test_data: The test set from Splitter class
		param_dist: Hyperparameters dictionary from which the best hyperparameters will be chosen
		search_size: The search size for hyperparameters (the number of random combinations to chechk)
		cv_size: The number of tests run of every cross validation
		parallel: Whether random search should be performed using multithreading
	"""

	def __init__(self, algorithm, training_data, test_data, param_dist, search_size, cv_size=3, parallel=False):
		"""
		Class initialisation.
		"""
		self._parallel = parallel
		self._training_data = training_data
		self._test_data = test_data
		self._algorithm = algorithm
		self._param_dist = param_dist
		self._search_size = search_size
		self._cv_size = cv_size
		self._tuned_alg = {}

	def _user_learn(self, args_x, args_y, vals_x, vals_y):
		"""
		For multithreading processing: single-user learn algorithm.

		Args:
			args_x: training features
			args_y: training targets
			vals_x: validation features
			vals_y: validation targets

		Returns:
			user id and score board with accuracy for each hyperparameters combination
		"""
		params_to_check = iterate_random_values(self._param_dist, self._search_size)
		score_board = {}
		for p_comb in params_to_check:
			score_board[tuple(sorted(p_comb.items()))] = []
			fold_avg = []
			for cv_fold in range(self._cv_size):
				alg_run = self._algorithm(**p_comb).fit(args_x, args_y)
				pred_run = alg_run.predict(vals_x)
				metric_val = accuracy_score(pred_run, vals_y)
				fold_avg.append(metric_val)
			metric_val = np.mean(fold_avg)
			score_board[tuple(sorted(p_comb.items()))].append(metric_val)
		ids = args_x.index.get_level_values(0)[0]
		return ids, score_board

	def learn(self):
		"""
		Learns the sklearn algorithm using cross-validation and passed input data.
		"""
		cnt = 0
		result_dic = {}
		for splits in self._training_data:  # for every cv split
			cnt += 1
			print("SPLIT: {}".format(cnt))
			train_x, train_y, val_x, val_y = splits
			if self._parallel:  # TODO: Finish result unpacking
				with cf.ThreadPoolExecutor(max_workers=6) as executor:
					args_x = [val for indi, val in train_x.groupby(level=0)]
					args_y = [val for indi, val in train_y.groupby(level=0)]
					vals_x = [val for indi, val in val_x.groupby(level=0)]
					vals_y = [val for indi, val in val_y.groupby(level=0)]
					results = list(
						tqdm(executor.map(self._user_learn, args_x, args_y, vals_x, vals_y), total=len(vals_y)))
				for result in results:
					if result[0] in result_dic.keys():
						result_dic[result[0]] += result[1]
					else:
						result_dic[result[0]] = result[1]
			else:  # single-threaded processing
				usrs = np.unique(train_x.index.get_level_values(0))
				for ids in tqdm(usrs, total=len(usrs)):
					params_to_check = iterate_random_values(self._param_dist, self._search_size)
					score_board = {}
					args_x = train_x.loc[ids]
					args_y = train_y.loc[ids]
					vals_x = val_x.loc[ids]
					vals_y = val_y.loc[ids]
					for p_comb in params_to_check:
						fold_avg = []
						for cv_fold in range(self._cv_size):
							alg_run = self._algorithm(**p_comb, n_jobs=-1).fit(args_x, args_y)
							pred_run = alg_run.predict(vals_x)
							metric_val = accuracy_score(pred_run, vals_y)
							fold_avg.append(metric_val)
						metric_val = np.mean(fold_avg)
						if tuple(sorted(p_comb.items())) in score_board.keys():
							score_board[tuple(sorted(p_comb.items()))].append(metric_val)
						else:
							score_board[tuple(sorted(p_comb.items()))] = [metric_val]
					if ids in result_dic.keys():
						for k, v in score_board.items():
							if k in result_dic[ids].keys():
								result_dic[ids][k].append(v)
							else:
								result_dic[ids][k] = [v]
					else:
						result_dic[ids] = {}
						for k, v in score_board.items():
							result_dic[ids][k] = [v]
		for usr, params in result_dic.items():  # selects the best params and trains the algorithm on them (for each user)
			select = {k: np.mean(v) for k, v in params.items()}
			best_params = dict(max(select.keys(), key=lambda x: select[x]))
			concat_x = pd.concat([self._training_data[-1][0].loc[usr], self._training_data[-1][2].loc[usr]])
			concat_y = pd.concat([self._training_data[-1][1].loc[usr], self._training_data[-1][3].loc[usr]])
			self._tuned_alg[usr] = self.algorithm(**best_params).fit(concat_x, concat_y)

	def test(self):
		"""
		Implements one and final algorithm test and saves the score.
		"""
		test_x, test_y = self._test_data
		usrs = np.unique(test_x.index.get_level_values(0))
		metrics = {}
		predictions_dic = {}
		for ids in tqdm(usrs, total=len(usrs)):
			cur_alg = self._tuned_alg[ids]
			preds = cur_alg.predict(test_x.loc[ids])
			metric_val = accuracy_score(preds, test_y.loc[ids])
			predictions_dic[ids] = preds
			metrics[ids] = metric_val
		self.scores = pd.Series(metrics)
		self.predictions = pd.concat([pd.concat({k: pd.Series(v) for k,v in predictions_dic.items()}).droplevel(1),test_y.droplevel(1)],axis=1)

	@property
	def algorithm(self):
		return self._algorithm


class Baseline():
	"""
	The baseline algorithm which assumes the next symbol be identical to the next one. Be aware that it uses information
	from the test set and thus, has advantage over other methods. Only test dataset is needed.

	Args:
		test_data: Test dataset. Do not use split data.
	"""

	def __init__(self, test_data):
		"""
		Class initialisation.
		"""
		self._test_data = test_data
		self.prediction = []

	def predict(self):
		"""
		Baseline prediction algorithm. This method only evaluates the accuracy of the method and do not return the
		predictions. Predictions are stored within the class attributes.

		Returns:
			accuracy score
		"""
		self.prediction = self._test_data[0].groupby(level=0).apply(lambda x: x.iloc[:, -1]).droplevel(1)  # make predictions
		aligned = pd.concat([self._test_data[1], self.prediction], axis=1)  # align predictions and true labels
		acc_score = aligned.groupby(level=0).apply(lambda x: sum(x.iloc[:, 0] == x.iloc[:, 1]) / x.shape[0])  # quickly evaluate the score
		print("SCORE: {}".format(acc_score.mean()))
		return aligned, acc_score


class TopLoc():
	"""
	This is a naive algorithm which assumes every symbol in the test dataset being the most frequently occurring symbol
	in the training data.

	Args:
		train_data: Train dataset, from the Splitter class. Do not use split data.
		test_data: Test dataset, from the Splitter class. Do not use split data.
	"""

	def __init__(self, train_data, test_data):
		"""
		Class initialisation.
		"""
		self._train_data = train_data
		self._test_data = test_data
		self.prediction = []

	def predict(self):
		"""
		Makes the prediction and returns the score. Predictions are stored within the class.

		Returns:
			Accuracy score
		"""
		tr_data = pd.concat([self._train_data[0][1], self._train_data[0][3]])
		top_location = tr_data.groupby(level=0).apply(lambda x: x.groupby(x).count().idxmax())
		to_conc = {}
		for uid, vals in self._test_data[1].groupby(level=0):
			vals[vals > -1] = top_location.loc[uid]
			to_conc[uid] = vals
		self.prediction = pd.concat(to_conc).droplevel(1)
		aligned = pd.concat([self._test_data[1], self.prediction], axis=1)
		acc_score = aligned.groupby(level=0).apply(lambda x: sum(x.iloc[:, 0] == x.iloc[:, 1]) / x.shape[0])
		print("SCORE: {}".format(acc_score.mean()))
		return aligned, acc_score


def split(trajectories_frame, train_size, state_size):
	"""
	Simple train-test data split for a Markov Chain.

	Args:
		trajectories_frame: TrajectoriesFrame class object
		train_size: The split ratio for training set
		state_size: The size of a window (for a Markov Chain algorithm)

	Returns:
		Split data
	"""
	train_frame = trajectories_frame['labels'].groupby(level=0).progress_apply(
		lambda x: x.iloc[:round(len(x) * train_size)])
	test_frame = trajectories_frame['labels'].groupby(level=0).progress_apply(
		lambda x: x.iloc[round(len(x) * train_size) - state_size:])
	return train_frame, test_frame


def markov_wrapper(trajectories_frame, test_size=.2, state_size=2, update=False, online=True):
	"""
	The wrapper, one stop shop algorithm for the Markov Chain. Splits the data, learns the model and makes predictions
	on the test set.

	Args:
		trajectories_frame: TrajectoriesFrame class object
		test_size: The training set size.
		state_size: The order of the Markov Chain
		update: Whether the model should update its beliefs based on own predictions
		averaged: Whether an averaged accuracy should be returned
		online: Whether the algorithm should make an online prediction (sees the last n symbols when predicting)

	Returns:
		Prediction scores
	"""
	train_size = 1 - test_size
	train_frame, test_frame = split(trajectories_frame, train_size, state_size)  # train test split
	test_lengths = test_frame.groupby(level=0).apply(lambda x: x.shape[0])
	predictions_dic = {}
	for uid, train_values in train_frame.groupby(level=0):  # training
		try:
			if online:
				predictions_dic[uid] = MarkovChain(list(train_values.values), state_size)
			else:
				predictions_dic[uid] = MarkovChain(list(train_values.values), state_size).move_from_build(
					test_lengths.loc[uid], update)
		except:
			continue
	results_dic = {}
	to_conc = {}
	for test_values, prediction_values in zip([g for g in test_frame.groupby(level=0)], predictions_dic):  # predicting
		uid = test_values[0]
		test_values = test_values[1].values
		if online:
			forecast = []
			for current_state in range(len(test_values) - state_size):
				forecast.append(predictions_dic[uid].move(test_values[current_state:current_state + state_size]))
			to_conc[uid] = forecast
			results_dic[uid] = sum(forecast == test_values[state_size:]) / len(forecast)
		else:
			results_dic[uid] = sum(test_values[state_size:] == predictions_dic[prediction_values]) / len(test_values)
			to_conc[uid] = predictions_dic[prediction_values]
	predictions = pd.DataFrame.from_dict(to_conc).unstack().droplevel(1)
	aligned = pd.concat([test_frame.droplevel([1,2]).groupby(level=0).apply(lambda x: x[state_size:]).droplevel([1]),predictions],axis=1)
	return aligned, pd.DataFrame.from_dict(results_dic,orient='index')


def sparse_wrapper(trajectories_frame, split_ratio=.8, state_size=0, update=False, averaged=True, online=False):
	"""
	"""
	train_frame, test_frame = split(trajectories_frame, split_ratio, state_size)
	test_lengths = test_frame.groupby(level=0).apply(lambda x: x.shape[0])
	predictions_dic = {}
	for uid, train_values in train_frame.groupby(level=0):
		predictions_dic[uid] = Sparse(train_values.values)
	results_dic = {}
	for test_values, prediction_values in zip([g for g in test_frame.groupby(level=0)], predictions_dic):  # predicting
		uid = test_values[0]
		test_values = test_values[1].values
		forecast = []
		split_ind = round(trajectories_frame.uloc(uid).shape[0]*split_ratio)
		for n in range(test_lengths.loc[uid]):
			context = trajectories_frame.uloc(uid).iloc[:split_ind].labels.values
			pred = predictions_dic[uid].predict(context)
			forecast.append(pred)
			split_ind += 1
		results_dic[uid] = sum(forecast == test_values[state_size:]) / len(forecast)
	if averaged:
		return sum(list(results_dic.values())) / len(results_dic.values())
	else:
		return results_dic
