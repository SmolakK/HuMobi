import numpy as np
import tqdm
from src.humobi.misc.utils import get_diags, normalize_chain, _equally_sparse_match, _equally_sparse_match_old, \
	remove_subset_rows, _equally_sparse_match_jit, sparse_predict_jit, unq_counts_model, sort_lines, cupy_sort_lines
from time import time
import concurrent.futures as cf
from itertools import repeat
from collections import deque
from numba import jit, prange
import pandas as pd


def normalize_list(l):
	suml = np.sum(l)
	return [x / suml for x in l]


def scale_vector(v):
	minim = np.min(v)
	maxim = np.max(v)
	if minim == maxim:
		return np.full_like(v, 0.5, dtype=np.float16)
	else:
		return (v - minim) / (maxim - minim)


class Sparse(object):
	"""
	Sparse predictor
	"""
	_search_size: int

	def __init__(self, search_size=None, reverse=False, overreach=False, rolls=True, remove_subsets=True,
	             reverse_overreach=True):
		self._search_size = search_size
		self.model = None
		self.reverse = reverse
		self.overreach = overreach
		self.reverse_overreach = reverse_overreach
		self.rolls = rolls
		self.remove_subsets = remove_subsets

	def _fit_multi(self, sequence, jit, max_search, n):
		nexts = deque([])
		matches = deque([])
		cur_id = len(sequence) - n
		if not self._search_size is None:
			start = cur_id - self._search_size
			start = np.clip(start, 0, None)
			end = cur_id + self._search_size
			end = np.clip(end, None, len(sequence))
		else:
			start = 0
		lookback = sequence[cur_id:end]
		search_space = sequence[start:cur_id]
		if jit:
			out = _equally_sparse_match_jit(lookback, search_space, overreach=self.overreach, roll=self.rolls)
		else:
			out = _equally_sparse_match(lookback, search_space, overreach=self.overreach, roll=self.rolls)
		if out and out[1].size != 0:
			if out[0].shape[1] > max_search:
				padded = out[0][:, -max_search:]
			else:
				padded = np.full((out[0].shape[0], max_search), -1)
				padded[:, -out[0].shape[1]:] = out[0]
			matches.append(padded)
			nexts.append(out[1])
		if self.reverse:
			if jit:
				out = _equally_sparse_match_jit(search_space, lookback, overreach=self.reverse_overreach,
				                                roll=self.rolls)
			else:
				out = _equally_sparse_match(search_space, lookback, overreach=self.reverse_overreach,
				                            roll=self.rolls)
			if out and out[1].size != 0:
				if out[0].shape[1] > max_search:
					padded = out[0][:, -max_search:]
				else:
					padded = np.full((out[0].shape[0], max_search), -1)
					padded[:, -out[0].shape[1]:] = out[0]
				matches.append(padded)
				nexts.append(out[1])
		if len(matches) == 0:
			matches = np.empty(max_search, dtype=np.int32)
			matches.fill(-1)
			nexts = np.array([-1])
		else:
			matches = np.vstack(matches)
			nexts = np.hstack(nexts)
		return matches, nexts

	def fit(self, sequence, jit=True, parallel=True, cuda=False, truncate=0.6):
		sequence = np.array(sequence, dtype=np.int64)
		sequence += 1  # REMEMBER     # TODO: flatten context/model
		nexts = deque([])
		matches = deque([])
		if self._search_size is None:
			max_search = len(sequence)
		else:
			max_search = self._search_size
		if parallel:
			with cf.ThreadPoolExecutor() as executor:
				all_cuts = list(range(1, len(sequence)))
				results = list(tqdm.tqdm(executor.map(self._fit_multi, repeat(sequence), repeat(jit),
				                                      repeat(max_search), all_cuts),
				                         total=len(sequence) - 1))
			matches = np.vstack([out[0] for out in results])
			nexts = np.hstack([out[1] for out in results])
		else:
			for n in tqdm.tqdm(range(1, len(sequence)), total=len(sequence) - 1):
				cur_id = len(sequence) - n
				if not self._search_size is None:
					start = cur_id - self._search_size
					start = np.clip(start, 0, None)
					end = cur_id + self._search_size
					end = np.clip(end, None, len(sequence))
				else:
					start = 0
				lookback = sequence[cur_id:end]
				search_space = sequence[start:cur_id]
				if jit:
					out = _equally_sparse_match_jit(lookback, search_space, overreach=self.overreach, roll=self.rolls)
				else:
					out = _equally_sparse_match(lookback, search_space, overreach=self.overreach, roll=self.rolls)
				if out and out[1].size != 0:
					padded = np.full((out[0].shape[0], max_search), -1)
					padded[:, -out[0].shape[1]:] = out[0]
					# padded = np.pad(out[0], ((0, 0), (max_search - out[0].shape[1], 0)), constant_values=-1)
					matches.append(padded)
					nexts.append(out[1])
				if self.reverse:
					if jit:
						out = _equally_sparse_match_jit(search_space, lookback, overreach=self.reverse_overreach,
						                                roll=self.rolls)
					else:
						out = _equally_sparse_match(search_space, lookback, overreach=self.reverse_overreach,
						                            roll=self.rolls)
					if out and out[1].size != 0:
						padded = np.full((out[0].shape[0], max_search), -1)
						padded[:, -out[0].shape[1]:] = out[0]
						# padded = np.pad(out[0], ((0, 0), (max_search - out[0].shape[1], 0)), constant_values=-1)
						matches.append(padded)
						nexts.append(out[1])
			nexts = np.hstack(nexts)
			matches = np.vstack(matches)

		# TRUNCATE
		if truncate is not None and truncate < 1:
			stacked_model = np.hstack((matches, nexts.reshape(-1, 1)))
			# unq_list, unq_counts = np.unique(stacked_model, return_counts=True, axis=0)
			stacked_model = cupy_sort_lines(stacked_model)
			unq_list, unq_counts = unq_counts_model(stacked_model)
			# stacked_unq = np.hstack((unq_counts.reshape(-1, 1), unq_list))
			# stacked_unq = stacked_unq[stacked_unq[:, 0].argsort()][::-1]
			# unq_list = stacked_unq[:, 1:]
			# unq_counts = stacked_unq[:, 0]
			# proportion = np.cumsum(unq_counts / np.sum(unq_counts))
			# filter_by_proportion = proportion > truncate
			# unq_list = unq_list[filter_by_proportion]
			# unq_counts = unq_counts[filter_by_proportion]
			# org_unq_weights = np.repeat(unq_counts, unq_counts, axis=0)
			# org_recency = np.repeat(org_recency, unq_counts, axis=0)
			# org_length = np.repeat(org_length, unq_counts, axis=0)
			# reconstructed = np.repeat(unq_list, unq_counts, axis=0)
			unq_list = unq_list[~(unq_list == -1).all(axis=1),:]
			matches = unq_list[:, :-1]
			nexts = unq_list[:, -1]

			flatmodel = matches + 1
			nonzero_elements = np.argwhere(np.fliplr(flatmodel))
			ind_first = np.unique(nonzero_elements[:, 0], return_index=True)[1]
			org_recency = nonzero_elements[ind_first, 1] + 1
			org_length = np.sum(np.clip(flatmodel, 0, 1), axis=1)
		# else:
		# unique_vals, unique_counts = unq_counts_model(matches,nexts)
		# unique_counts = np.array(unique_counts)
		# unique_rows = np.array(unique_vals)
		# org_unq_weights = np.repeat(unique_counts, unique_counts)
		# rebuilt = np.repeat(unique_rows, unique_counts, axis=0)
		# matches = rebuilt[:, :-1]
		# nexts = rebuilt[:, -1]

		if self.remove_subsets:
			stacks = remove_subset_rows(np.hstack((matches, nexts[:, np.newaxis])))
			self.model = (stacks[:, :-1], stacks[:, -1])
		else:
			self.model = (matches, nexts, unq_counts, org_recency,
			              org_length)  # matches, nexts, counts-compression, recency, lengths - all compressed)
		# self.model = (matches, nexts)

	def predict(self, context, recency_weights=None, length_weights=None, from_dist=False,
	            org_recency_weights=None, org_length_weights=None, jit=True, completeness_weights=None,
	            uniqueness_weights=None):

		if jit:
			prediction = sparse_predict_jit(context, self.model[0], self.model[1], self.model[2], recency_weights,
			                                length_weights,
			                                from_dist,
			                                org_recency_weights, org_length_weights,
			                                completeness_weights=completeness_weights,
			                                uniqueness_weights=uniqueness_weights)
		else:
			prediction = self._predict(context, recency_weights, length_weights, from_dist,
			                           org_recency_weights, org_length_weights,
			                           completeness_weights=completeness_weights,
			                           uniqueness_weights=uniqueness_weights)
		return prediction

	def _predict(self, context, recency_weights=None, length_weights=None, from_dist=False,
	             org_recency_weights=None, org_length_weights=None, completeness_weights=None, uniqueness_weights=None,
	             count_weights=None):
		# TODO: Weights separated (wrap) and compressed model
		# new weighs: completeness: DONE, uniqueness,
		# ideas: separate weighing,
		# separate truncate if goood,
		# independent weights: DONE
		model_size = self.model[0].shape[1]
		pad_size = model_size - context.shape[0]
		if pad_size > 0:
			context = np.pad(context, (pad_size, 0))
		elif pad_size < 0:
			context = context[-model_size:]

		model = self.model[0]
		candidates = self.model[1]
		counts = self.model[2]
		recency = self.model[3]
		lengths = self.model[4]

		# MATCHING
		matches = (model == context)
		match_mask = np.sum(matches, axis=1) >= 1
		if np.sum(match_mask) == 0:
			unq = np.unique(self.model[1], return_counts=True)
			p = unq[1] / np.sum(unq[1])
			SMC = unq[0][np.argmax(p)]
			counte = pd.DataFrame(unq[1] / sum(unq[1]))
			counte.index = unq[0]
			return SMC, counte.T.to_dict(orient='index')

		# Weights
		matches = matches[match_mask]
		candidates = candidates[match_mask]
		counts = counts[match_mask]
		recency = recency[match_mask]
		lengths = lengths[match_mask]

		matches_sum = np.sum(matches, axis=1)
		if uniqueness_weights is not None:
			unq_weights = self.weighting_reversed(counts, uniqueness_weights)
		else:
			unq_weights = np.ones_like(matches_sum)

		if org_recency_weights is not None:
			org_recency = self.weighting_reversed(recency, org_recency_weights)
		else:
			org_recency = np.ones_like(matches_sum)

		if org_length_weights is not None:
			org_lengths = self.weighting(lengths, org_length_weights)
		else:
			org_lengths = np.ones_like(matches_sum)

		if completeness_weights is not None:
			completeness_measure = matches_sum/lengths
			completeness_weights = self.weighting(completeness_measure, completeness_weights)
		else:
			completeness_weights = np.ones_like(matches_sum)

		if count_weights is not None:
			count_weights = self.weighting(counts,count_weights)
		else:
			count_weights = np.ones_like(matches_sum)

		if recency_weights is not None:
			recency = self.weight_recency(matches, recency_weights)
		else:
			recency = np.ones_like(matches_sum)

		if length_weights is not None:
			lengths = self.weight_length(matches_sum, length_weights)
		else:
			lengths = np.ones_like(matches_sum)

		matches_sum = np.multiply(matches_sum, org_recency)
		matches_sum = np.multiply(matches_sum, org_lengths)
		matches_sum = np.multiply(matches_sum, recency)
		matches_sum = np.multiply(matches_sum, lengths)
		matches_sum = np.multiply(matches_sum, completeness_weights)
		matches_sum = np.multiply(matches_sum, unq_weights)
		matches_sum = np.multiply(matches_sum, count_weights)

		# PREDICTION
		joined = np.vstack([matches_sum.T, candidates]).T
		joined = joined[joined[:, 1].argsort()]
		spliter = np.unique(joined[:, 1], return_index=True)
		joined = np.split(joined[:, 0], spliter[1][1:])
		probs = [np.sum(x) for x in joined]
		probs = probs / sum(probs)
		if from_dist:
			SMC = np.random.choice(spliter[0], p=probs)
		else:
			SMC = spliter[0][np.argmax(probs)]
		probs = pd.DataFrame(probs)
		probs.index = spliter[0]
		return SMC, probs.T.to_dict(orient='index')

	def weight_recency(self, vect, recency_weights):
		nonzero_elements = np.argwhere(np.fliplr(vect))
		ind_first = np.unique(nonzero_elements[:, 0], return_index=True)[1]
		last_nonzero = nonzero_elements[ind_first, 1] + 1
		if recency_weights in ['inverted', 'IW']:
			recency_func = lambda x: 1 / x
			recency = np.array(list(map(recency_func, last_nonzero)))
		elif recency_weights in ['inverted squared', 'IWS']:
			recency_func = lambda x: 1 / x ** 2
			recency = np.array(list(map(recency_func, last_nonzero)))
		elif recency_weights in ['inverted qubic', 'IWQ']:
			recency_func = lambda x: 1 / x ** 3
			recency = np.array(list(map(recency_func, last_nonzero)))
		elif recency_weights in ['linear', 'quadratic', 'L', 'Q']:
			last_nonzero = self.model[0].shape[1] - last_nonzero + 1
			if recency_weights in ['linear', 'L']:
				recency = last_nonzero / self.model[0].shape[1]
			else:
				recency = (last_nonzero / self.model[0].shape[1]) ** 2
		return recency

	def weight_length(self, vect, length_weights):
		if length_weights in ['inverted', 'IW']:
			weights_func = lambda x: 1 / x
		elif length_weights in ['inverted squared', 'IWS']:
			weights_func = lambda x: 1 / x ** 2
		elif length_weights in ['linear', 'L']:
			weights_func = lambda x: x
		elif length_weights in ['quadratic', 'Q']:
			weights_func = lambda x: x ** 2
		lengths = np.array(list(map(weights_func, vect)))
		lengths = scale_vector(lengths)
		return lengths

	def weighting(self, vect, weights):
		if weights in ['inverted', 'IW']:
			weights_func = lambda x: 1 / x
		elif weights in ['inverted squared', 'IWS']:
			weights_func = lambda x: 1 / (x ** 2)
		elif weights in ['linear', 'L']:
			weights_func = lambda x: x
		elif weights in ['quadratic', 'Q']:
			weights_func = lambda x: x ** 2
		elif weights in ['flat', 'F']:
			weights_func = lambda x: x
		lengths = np.array(list(map(weights_func, vect)))
		if weights not in ['flat', 'F']:
			lengths = scale_vector(lengths)
		return lengths

	def weighting_reversed(self, vect, weights):
		if weights in ['inverted', 'IW']:
			weights_func = lambda x: 1 / x
			unqs = np.array(list(map(weights_func, vect)))
			unqs = scale_vector(unqs)
		elif weights in ['inverted squared', 'IWS']:
			weights_func = lambda x: 1 / x ** 2
			unqs = np.array(list(map(weights_func, vect)))
			unqs = scale_vector(unqs)
		elif weights in ['inverted qubic', 'IWQ']:
			weights_func = lambda x: 1 / x ** 3
			unqs = np.array(list(map(weights_func, vect)))
			unqs = scale_vector(unqs)
		elif weights in ['linear', 'L']:
			vect_reversed = (vect.max() - vect) + 1
			unqs = scale_vector(vect_reversed)
		elif weights in ['quadratic', 'Q']:
			vect_reversed = (vect.max() - vect) + 1
			unqs = scale_vector(vect_reversed ** 2)
		return unqs


class Sparse_old(object):
	"""
	Sparse predictor
	"""

	def __init__(self):
		self.model = None

	def fit(self, sequence):
		matches = []
		nexts = []
		for n in tqdm.tqdm(range(1, len(sequence)), total=len(sequence) - 1):
			cur_id = len(sequence) - n
			if cur_id > 0:
				lookback = sequence[cur_id:]
				search_space = sequence[:cur_id]
			out = _equally_sparse_match_old(lookback, search_space)
			if out:
				matches.append(np.stack([x[0] for x in out]))
				nexts.append(np.stack([x[1] for x in out]))
			out = _equally_sparse_match_old(search_space, lookback)
			if out:
				matches.append(np.stack([x[0] for x in out]))
				nexts.append(np.stack([x[1] for x in out]))
		matches = np.vstack(matches)
		nexts = np.hstack(nexts)
		self.model = (matches, nexts)

	def predict(self, context, recency_weights=None, length_weights=None, from_dist=False):
		# TODO: matches length original, recency original
		model_size = self.model[0].shape[1]
		pad_size = model_size - context.shape[0]
		if pad_size > 0:
			context = np.pad(context[0], (pad_size, 0))
		elif pad_size < 0:
			context = context[-model_size:]
		matches = (self.model[0] == context)
		match_mask = np.sum(matches, axis=1) >= 1
		# RECENCY
		if recency_weights in ['inverted', 'inverted squared', 'IW', 'IWS']:
			nonzero_elements = np.argwhere(np.fliplr(matches))
			ind_first = np.unique(nonzero_elements[:, 0], return_index=True)[1]
			last_nonzero = nonzero_elements[ind_first, 1] + 1
			if recency_weights in ['inverted', 'IW']:
				recency_func = lambda x: 1 / x
			else:
				recency_func = lambda x: 1 / x ** 2
			recency = np.array(list(map(recency_func, last_nonzero)))
		elif recency_weights in ['linear', 'quadratic', 'L', 'Q']:
			nonzero_elements = np.argwhere(np.fliplr(matches))
			ind_first = np.unique(nonzero_elements[:, 0], return_index=True)[1]
			last_nonzero = nonzero_elements[ind_first, 1] + 1
			last_nonzero = self.model[0].shape[1] - last_nonzero + 1
			if recency_weights in ['linear', 'L']:
				recency = last_nonzero / model_size
			else:
				recency = (last_nonzero / model_size) ** 2
		else:
			recency = np.ones(np.sum(match_mask))
		# LENGTHS
		matches = np.sum(matches, axis=1)
		matches = matches[match_mask]
		candidates = self.model[1][match_mask]
		if length_weights is not None:
			if length_weights in ['inverted', 'IW']:
				weights_func = lambda x: 1 / x
			elif length_weights in ['inverted squared', 'IWS']:
				weights_func = lambda x: 1 / x ** 2
			elif length_weights in ['linear', 'L']:
				weights_func = lambda x: x
			elif length_weights in ['quadratic', 'Q']:
				weights_func = lambda x: x ** 2
			lengths = np.array(list(map(weights_func, matches)))
			lengths = scale_vector(lengths)
			matches = np.multiply(matches, lengths)
		matches = np.multiply(matches, recency)
		joined = np.vstack([matches.T, candidates]).T
		joined = joined[joined[:, 1].argsort()]
		spliter = np.unique(joined[:, 1], return_index=True)
		joined = np.split(joined[:, 0], spliter[1][1:])
		probs = [np.sum(x) for x in joined]
		probs = probs / sum(probs)
		if from_dist:
			SMC = np.random.choice(spliter[0], p=probs)
		else:
			SMC = spliter[0][np.argmax(probs)]
		probs = pd.DataFrame(probs)
		probs.index = spliter[0]
		return SMC, probs.T.to_dict(orient='index')
