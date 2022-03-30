import numpy as np
import sys

sys.path.append("..")
from misc.utils import to_labels
from tqdm import tqdm

tqdm.pandas()


def normalize_chain(dicto):
	"""
	Normalizes dictionary values. Used for the Markov Chain normalization.
	:param dicto: dictionary to normalize
	:return: normalized dictionary
	"""
	total = 1 / float(np.sum(list(dicto.values())))
	for k, v in dicto.items():
		dicto[k] = v * total
	return dicto


def build_single(sequence, state_size, state):
	"""
	A small temporary model for lower order Markov Chains called during prediction when previously unseen state is
	encountered.
	:param sequence: Sequence to learn
	:param state_size: Order of the Markov Chain
	:param state: Current state
	:return: Lower order Markov Chain
	"""
	model = {}
	for posnum in range(len(sequence) - state_size):
		if tuple(sequence[posnum:posnum + state_size]) == state:
			next = sequence[posnum + state_size]
			if state not in model.keys():
				model[state] = {}
			if next not in model[state].keys():
				model[state][next] = 0
			model[state][next] += 1
	return model


class MarkovChain(object):
	"""
	Markov Chain class.
	:param sequence: Sequence which is used for chain building
	:param state_size: The order of the Markov Chain
	"""

	def __init__(self, sequence, state_size):
		"""
		Class initialization. Calls chain building (learning).
		"""
		self._state_size = state_size
		self._sequence = sequence
		self.model = self.build()

	@property
	def state_size(self):
		return self._state_size

	@property
	def sequence(self):
		return self._sequence

	@sequence.setter
	def sequence(self, value):
		self._sequence = value

	def build(self):
		"""
		Builds the Markov Chain. Returned model is not normalized as it is normalized during prediction.
		:return: Markov Model
		"""
		model = {}  # the model is a dictionary
		for posnum in range(len(self.sequence) - self.state_size):  # for each element in the sequence
			state = tuple(self.sequence[posnum:posnum + self.state_size])  # read current state, including order
			next = self.sequence[posnum + self.state_size]  # read the next symbol
			if state not in model.keys():  # if symbol not yet encountered
				model[state] = {}  # create a slot for it
			if next not in model[state].keys():  # if symbol encountered but the next state haven't been encountered yet for that symbol
				model[state][next] = 0  # create a slot for it
			model[state][next] += 1  # count +1 for that transition
		return model

	def move(self, state):
		"""
		Predict the next symbol based on the given state.
		:param state: The state from which the Markov Chain will make prediction
		:return: Predicted symbol
		"""
		state = tuple(state)  # current state - the model operates on tuples
		if state not in self.model.keys():  # this whole if operates on the case when the state is not met and fits lower-order models
			for lower in range(1, self.state_size + 1):
				lower_state = state[lower::]
				lower_state_size = self.state_size - lower
				lower_state_sequence = [tuple(self.sequence[posnum:posnum + lower_state_size]) \
				                        for posnum in range(len(self.sequence) - lower_state_size)]
				if lower_state in lower_state_sequence:
					break
				if len(lower_state) == 0:
					return -1
			temp_model = build_single(self.sequence, lower_state_size, lower_state)  # builds temporal, smaller model
			transit = normalize_chain(temp_model[lower_state])
		else:
			transit = normalize_chain(self.model[state])  # normalize chain
		prediction = np.random.choice(list(transit.keys()), p=list(transit.values()))  # make prediction
		return prediction

	def move_from_build(self, horizon, update=False):
		"""
		Predict the next symbols based on the last seen state during training
		:param horizon: How many symbols to predict
		:param update: Whether the model should be updated by the new predictions (not recommended)
		:return: Predicted symbols
		"""
		predicted_sequence = []
		recent_state = self.sequence[-self.state_size:]
		for steps in range(horizon):
			prediction = self.move(recent_state)
			predicted_sequence.append(prediction)
			recent_state.append(prediction)
			recent_state = recent_state[1:]
			if update:
				self.sequence += prediction
				self.build()
		return np.array(predicted_sequence)
