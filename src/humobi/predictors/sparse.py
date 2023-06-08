import numpy as np
import tqdm
from src.humobi.misc.utils import get_diags, normalize_chain, _equally_sparse_match, _equally_sparse_match_old, \
	remove_subset_rows, _equally_sparse_match_jit
from time import time
import concurrent.futures as cf
from itertools import repeat
from collections import deque
from numba import jit, prange

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

	def _fit_multi(self,sequence, jit, max_search,n):
		nexts = deque([])
		matches = deque([])
		cur_id = len(sequence) - n
		if not self._search_size is None:
			start = cur_id - self._search_size
			start = np.clip(start, 0, None)
			end = cur_id + self._search_size
			end = np.clip(end,None,len(sequence))
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
				padded = out[0][:,-max_search:]
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
			matches = np.empty(max_search,dtype=np.int32)
			matches.fill(-1)
			nexts = np.array([-1])
		else:
			matches = np.vstack(matches)
			nexts = np.hstack(nexts)
		return matches,nexts

	def fit(self, sequence, jit = True, parallel = True, cuda = False):
		sequence = np.array(sequence)
		sequence += 1  # REMEMBER
		nexts = deque([])
		matches = deque([])
		if self._search_size is None:
			max_search = len(sequence)
		else:
			max_search = self._search_size
		if parallel:
			with cf.ThreadPoolExecutor() as executor:
				all_cuts = list(range(1,len(sequence)))
				results = list(tqdm.tqdm(executor.map(self._fit_multi,repeat(sequence), repeat(jit),
				                                      repeat(max_search),all_cuts),
				                         total=len(sequence)-1))
			matches = np.vstack([out[0] for out in results])
			nexts = np.hstack([out[1] for out in results])
		else:
			for n in tqdm.tqdm(range(1, len(sequence)), total=len(sequence) - 1):
				cur_id = len(sequence) - n
				if not self._search_size is None:
					start = cur_id - self._search_size
					start = np.clip(start, 0, None)
				else:
					start = 0
				lookback = sequence[cur_id:]
				search_space = sequence[start:cur_id]
				if jit:
					out = _equally_sparse_match_jit(lookback, search_space, overreach=self.overreach, roll=self.rolls)
				else:
					out = _equally_sparse_match(lookback, search_space, overreach=self.overreach, roll=self.rolls)
				if out and out[1].size != 0:
					padded = np.full((out[0].shape[0],max_search),-1)
					padded[:, -out[0].shape[1]:] = out[0]
					# padded = np.pad(out[0], ((0, 0), (max_search - out[0].shape[1], 0)), constant_values=-1)
					matches.append(padded)
					nexts.append(out[1])
				if self.reverse:
					if jit:
						out = _equally_sparse_match_jit(search_space, lookback, overreach=self.reverse_overreach, roll=self.rolls)
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
		if self.remove_subsets:
			stacks = remove_subset_rows(np.hstack((matches, nexts[:, np.newaxis])))
			self.model = (stacks[:, :-1], stacks[:, -1])
		else:
			self.model = (matches, nexts)

	def predict(self, context, recency_weights=None, length_weights=None, from_dist=False,
	            org_recency_weights=None, org_length_weights=None, jit = True):
		return self._predict(context, recency_weights, length_weights, from_dist,
			                org_recency_weights, org_length_weights)

	def _predict(self, context, recency_weights=None, length_weights=None, from_dist=False,
	            org_recency_weights=None, org_length_weights=None):
		model_size = self.model[0].shape[1]
		pad_size = model_size - context.shape[0]
		if pad_size > 0:
			context = np.pad(context, (pad_size, 0))
		elif pad_size < 0:
			context = context[-model_size:]
		matches = (self.model[0] == context)
		match_mask = np.sum(matches, axis=1) >= 1
		if np.sum(match_mask) == 0:
			unq = np.unique(self.model[1], return_counts=True)
			SMC = np.random.choice(unq[0], p=unq[1] / np.sum(unq[1]))
			return SMC

		# Weights
		matches = matches[match_mask]
		matches_sum = np.sum(matches, axis=1)
		if org_recency_weights is not None and org_length_weights is not None:
			flatmodel = (self.model[0]+1)[match_mask]
			if org_recency_weights is not None:
				org_recency = self.weight_recency(flatmodel, org_recency_weights)
				matches_sum = np.multiply(matches_sum, org_recency)
			if org_length_weights is not None:
				vect_org_lengths = np.sum(np.clip(flatmodel,0,1), axis=1)
				org_lengths = self.weight_length(vect_org_lengths, org_length_weights)
				matches_sum = np.multiply(matches_sum, org_lengths)
		if recency_weights is not None:
			recency = self.weight_recency(matches, recency_weights)
			matches_sum = np.multiply(matches_sum, recency)
		if length_weights is not None:
			lengths = self.weight_length(matches_sum, length_weights)
			matches_sum = np.multiply(matches_sum, lengths)

		# PREDICTION
		candidates = self.model[1][match_mask]
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
		return SMC

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
		return SMC
