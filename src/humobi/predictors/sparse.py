import numpy as np
import tqdm
from src.humobi.misc.utils import get_diags, normalize_chain, _equally_sparse_match, _equally_sparse_match_old, remove_subset_rows
from time import time

def normalize_list(l):
	suml = np.sum(l)
	return [x/suml for x in l]


def scale_vector(v):
	return (v-np.min(v))/(np.max(v)-np.min(v))


class Sparse(object):
	"""
	Sparse predictor
	"""
	_search_size: int

	def __init__(self, search_size = None, reverse = False, overreach = False, rolls = True, remove_subsets = True):
		self._search_size = search_size
		self.model = None
		self.reverse = reverse
		self.overreach = overreach
		self.rolls = rolls
		self.remove_subsets = remove_subsets

	def fit(self, sequence):
		sequence = np.array(sequence)
		sequence += 1 #REMEBER
		nexts = []
		max_search = len(sequence)
		matches = np.zeros((0,max_search))
		for n in tqdm.tqdm(range(1, len(sequence)),total=len(sequence)-1):
			cur_id = len(sequence) - n
			if not self._search_size is None:
				start = self._search_size+n
			else:
				start = 0
			lookback = sequence[cur_id:]
			search_space = sequence[start:cur_id]
			out = _equally_sparse_match(lookback, search_space, overreach=self.overreach, roll = self.rolls)
			if out[1].size != 0:
				padded = np.pad(out[0],((0,0),(max_search-out[0].shape[1],0)),constant_values=-1)
				matches = np.append(matches,padded,axis=0)
				nexts.append(out[1])
			if self.reverse:
				out = _equally_sparse_match(search_space, lookback, overreach=self.overreach, roll = self.rolls)
				if out[1].size != 0:
					padded = np.pad(out[0],((0,0),(max_search-out[0].shape[1],0)),constant_values=-1)
					matches = np.append(matches, padded, axis=0)
					nexts.append(out[1])
		nexts = np.hstack(nexts)
		if self.remove_subsets:
			stacks = remove_subset_rows(np.hstack((matches,nexts[:,np.newaxis])))
			self.model = (stacks[:,:-1],stacks[:,-1])
		else:
			self.model = (matches,nexts)

	def predict(self, context, recency_weights=None, length_weights=None, from_dist = False):
		#TODO: matches length original, recency original
		model_size = self.model[0].shape[1]
		pad_size = model_size - context.shape[0]
		if pad_size > 0:
			context = np.pad(context, (pad_size, 0)) #TODO:??
		elif pad_size < 0:
			context = context[-model_size:]
		matches = (self.model[0] == context)
		match_mask = np.sum(matches,axis=1) >= 1
		#RECENCY
		if recency_weights in ['inverted','inverted squared','IW','IWS']:
			nonzero_elements = np.argwhere(np.fliplr(matches))
			ind_first = np.unique(nonzero_elements[:,0],return_index=True)[1]
			last_nonzero = nonzero_elements[ind_first,1]+1
			if recency_weights in ['inverted','IW']:
				recency_func = lambda x: 1/x
			else:
				recency_func = lambda x: 1/x**2
			recency = np.array(list(map(recency_func, last_nonzero)))
		elif recency_weights in ['linear','quadratic','L','Q']:
			nonzero_elements = np.argwhere(np.fliplr(matches))
			ind_first = np.unique(nonzero_elements[:, 0], return_index=True)[1]
			last_nonzero = nonzero_elements[ind_first, 1] + 1
			last_nonzero = self.model[0].shape[1] - last_nonzero + 1
			if recency_weights in ['linear','L']:
				recency = last_nonzero/model_size
			else:
				recency = (last_nonzero/model_size)**2
		else:
			recency = np.ones(np.sum(match_mask))
		#LENGTHS
		matches = np.sum(matches, axis=1)
		matches = matches[match_mask]
		candidates = self.model[1][match_mask]
		if length_weights is not None:
			if length_weights in ['inverted','IW']:
				weights_func = lambda x: 1/x
			elif length_weights in ['inverted squared','IWS']:
				weights_func = lambda x: 1/x**2
			elif length_weights in ['linear','L']:
				weights_func = lambda x: x
			elif length_weights in ['quadratic','Q']:
				weights_func = lambda x: x**2
			lengths = np.array(list(map(weights_func, matches)))
			lengths = scale_vector(lengths)
			matches = np.multiply(matches,lengths)
		matches = np.multiply(matches, recency)
		joined = np.vstack([matches.T, candidates]).T
		joined = joined[joined[:,1].argsort()]
		spliter = np.unique(joined[:,1], return_index=True)
		joined = np.split(joined[:,0],spliter[1][1:])
		probs = [np.sum(x) for x in joined]
		if sum(probs) == 0:
			unq = np.unique(self.model[1],return_counts=True)
			SMC = np.random.choice(unq[0],p=unq[1]/np.sum(unq[1]))
		else:
			probs = probs / sum(probs)
			if from_dist:
				SMC = np.random.choice(spliter[0], p=probs)
			else:
				SMC = spliter[0][np.argmax(probs)]
		return SMC


class Sparse_old(object):
	"""
	Sparse predictor
	"""

	def __init__(self):
		self.model = None

	def fit(self, sequence):
		scanthrough = {}
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
		self.model = (matches,nexts)

	def predict(self, context, recency_weights=None, length_weights=None, from_dist = False):
		#TODO: matches length original, recency original
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