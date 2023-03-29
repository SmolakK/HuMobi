import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from numba import cuda, jit
from math import ceil
from Bio import pairwise2

def normalize_chain(dicto):
	"""
	Normalizes dictionary values. Used for the Markov Chain normalization.

	Args:
		dicto: dictionary to ..

	Returns:
		..d dictionary
	"""
	total = 1 / float(np.sum(list(dicto.values())))
	for k, v in dicto.items():
		dicto[k] = v * total
	return dicto


def get_diags(a):
	"""
	Extracts all the diagonals from the matrix.

	Args:
		a: numpy array to process

	Returns:
		a list of diagonals
	"""
	diags = [a.diagonal(i) for i in range(-a.shape[0] + 1, a.shape[1])]
	return [n.tolist() for n in diags]


def resolution_to_points(r, c_max, c_min):
	"""
	Calculates how many points are needed to divide range for given resolution
	:param r: resolution
	:param c_max: maximum value
	:param c_min: minimum value
	:return: the number of points
	"""
	c_range = c_max - c_min
	c_points = c_range / r + 1
	return ceil(c_points)


def moving_average(a, n=2):
	"""
	Implements fast moving average
	:param a: input data array
	:param n: window size
	:return: processed data array
	"""
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


def _matchfinder(start_idx, gs, data_len):
	"""
	Finds the shortest not repeating sequences according to the Lempel-Ziv algorithm
	:param start_idx: starting point in the array from which search will be started
	:param gs: symbol series
	:param data_len: data length
	:return: current starting index and the shortest non-repeating subsequence length
	"""
	max_subsequence_matched = 0
	for i in range(0, start_idx):
		j = 0
		end_distance = data_len - start_idx
		while (start_idx + j < data_len) and (i + j < start_idx) and (gs[i + j] == gs[start_idx + j]):
			j += 1
		if j == end_distance:
			return start_idx, 0
		elif j > max_subsequence_matched:
			max_subsequence_matched = j
	return start_idx, max_subsequence_matched + 1


@cuda.jit
def _matchfinder_gpu(gs, data_len, output):
	"""
	Finds the shortest not repeating sequences according to the Lempel-Ziv algorithm. Algorithm adaptation for GPU.
	:param gs: symbol series
	:param data_len: data length
	:param output: output array
	"""
	pos = cuda.grid(1)
	max_subsequence_matched = 0
	finish_bool = False
	if pos < data_len:
		for i in range(0, pos):
			j = 0
			end_distance = data_len - pos
			while (pos + j < data_len) and (i + j < pos) and (gs[i + j] == gs[pos + j]):
				j += 1
			if j == end_distance:
				finish_bool = True
				break
			elif j > max_subsequence_matched:
				max_subsequence_matched = j
		if finish_bool:
			output[pos] = end_distance + 1  # CHANGED with XU paper
		else:
			output[pos] = max_subsequence_matched + 1


def matchfinder(gs, gpu=True):
	"""
	Finds the shortest not repeating sequences according to the Lempel-Ziv algorithm
	:param gs: symbol series
	:return: the length of the shortest non-repeating subsequences at each step of sequence
	"""
	gs = gs.dropna()
	data_len = len(gs)
	gs = np.array(gs.values)
	output = np.zeros(data_len)
	output[0] = 1
	if gpu:
		threadsperblock = 256
		blockspergrid = ceil(data_len / threadsperblock)
		_matchfinder_gpu[threadsperblock, blockspergrid](gs, data_len, output)
	return output


@jit
def _repeatfinder_dense(s1, s2):
	output = np.zeros(len(s2))
	for pos1 in range(0, len(s2)):
		max_s = 0
		for pos2 in range(0, len(s1)):
			j = 0
			while s1[pos2 + j] == s2[pos1 + j]:
				j += 1
				if pos2 + j == len(s1) or pos1 + j == len(s2):
					break
			if j > max_s:
				max_s = j
		output[pos1] = max_s
	return max(output)-1 / len(s2)-1


@jit
def _repeatfinder_sparse(s1, s2):
	matrix = [[0 for x in range(len(s2))] for x in range(len(s1))]
	for i in range(len(s1)):
		for j in range(len(s2)):
			if s1[i] == s2[j]:
				if i == 0 or j == 0:
					matrix[i][j] += 1
				else:
					matrix[i][j] = matrix[i - 1][j - 1] + 1
			else:
				matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1])
	cs = matrix[-1][-1]
	return cs-1 / len(s2)-1


def _repeatfinder_equally_sparse(s1, s2):
	matrix = np.zeros((len(s1), len(s2)))  # prepare matrix for results
	for i in range(len(s1)):
		for j in range(len(s2)):
			if s1[i] == s2[j]:
				if i == 0 or j == 0:  # if matched symbols are at the start of the sequence
					matrix[i][j] += 1  # if symbols matched - add 1
				else:
					matrix[i][j] = 1  # if symbols matched - add 1
	s2_indi = (np.vstack(
		[np.arange(matrix.shape[1]) for x in range(matrix.shape[0])]) + 1) * matrix  # convert matched 1's into indices
	s2diags = get_diags(s2_indi)  # get all diagonals
	nonzero_transitions = [[y for y in x if y != 0] for x in s2diags]  # filter out empty lists
	nonzero_transitions = [[(x[y], x[y + 1]) for y in range(len(x) - 1)] for x in
	                       nonzero_transitions]  # convert to transitions
	translateds2 = [[(s2[int(y[0]) - 1], s2[int(y[1]) - 1]) for y in x] for x in nonzero_transitions if
	                len(x) > 0]  # convert to matched symbols
	# search for self-transitions and transitions
	try:
		selft = sum([x[0] == x[1] for x in sorted(translateds2, key=len)[-1]])
	except:
		selft = 0
	try:
		nonselft = sum([x[0] != x[1] for x in sorted(translateds2, key=len)[-1]])
	except:
		nonselft = 0
	return sum([selft, nonselft]) / (len(s2) - 1)


def _global_align(s1, s2):
	one = list(s1)
	two = list(s2)
	alignment = \
		pairwise2.align.globalms(one, two, 1, -1, -1, 0, penalize_end_gaps=False, one_alignment_only=True,
		                         gap_char=['-'])[
			0]
	return alignment.score / (len(two)-1)


def _iterative_global_align(s1, s2):
	one = list(s1)  # preapare lists of symbols
	two = list(s2)
	cut = two  # assign currently processed sequence
	all_match = []
	to_search = []
	while True:
		best_match = \
			pairwise2.align.globalms(one, cut, 1, -1, -1, 0, penalize_end_gaps=False,
			                         one_alignment_only=True, gap_char=['-'])[
				0]
		zipped = [(x, y) for x, y in zip(best_match[0], best_match[1])]  # combinations of matched symbols
		out_of_match = [1 if x[0] != x[1] and x[0] == '-' else 0 for x in zipped]  # search for mismatched symbols
		out_diff = np.diff(out_of_match)  # find gaps
		starts = [x for x in range(len(out_diff)) if out_diff[x] == 1]  # find starts of gaps
		ends = [x for x in range(len(out_diff)) if out_diff[x] == -1]  # find ends of gaps
		lengths = [y - x for x, y in zip(starts, ends)]  # find lengths of gaps
		first_ends = [1 if x == '-' else 0 for x in best_match[0]]  # find mismatched starts and ends of sequence
		first_ends_diff = np.diff(first_ends)  # find range of mismatched starts and ends
		first_ends_starts = [x for x in range(len(first_ends_diff)) if first_ends_diff[x] == 1]
		first_ends_ends = [x for x in range(len(first_ends_diff)) if first_ends_diff[x] == -1]
		if len(first_ends_ends) > 0 and 0 not in first_ends[:first_ends_ends[0]]:
			begin = best_match[1][:first_ends_ends[0] + 1]
		else:
			begin = []
		if len(first_ends_starts) > 0 and 0 not in first_ends[first_ends_starts[-1] + 1:]:
			end = best_match[1][first_ends_starts[-1] + 1:]
		else:
			end = []
		lengths += [len(begin), len(end)]
		maxleng = np.max(lengths)  # check which mismatch is the longest
		longest = np.where(lengths == maxleng)  # take the mismatched part
		if longest[0][0] == len(lengths) - 1:  # check if its a gap in the middle or at the ends
			to_search.append(end)
		elif longest[0][0] == len(lengths) - 2:
			to_search.append(begin)
		else:
			for n in longest[0]:
				n = n
				if n == len(lengths) - 1:
					to_search.append(end)
				elif n == len(lengths) - 2:
					to_search.append(begin)
				else:
					to_search.append(best_match[1][starts[n] + 1:ends[n] + 1])
		if len(to_search) == 0:  # if there are no sequences to match - stop
			if best_match[2] > 0:  # if there was positive score in the last match - add it to the list
				all_match.append(best_match[2] - 1)  # add score to the list of scores (-1 for transitions)
			break
		cut = to_search.pop(0)  # pop out the sequence for search
		cut = [x for x in cut if isinstance(x, float)]  # take only symbols
		if best_match[2] <= 0:  # if there is zero score already - stop
			break
		all_match.append(best_match[2] - 1)  # add score to the list of scores (-1 for transitions)
		if len(cut) <= 1:
			break  # if sequence for search does not consist of at least two symbols - stop
	return sum(all_match) / (len(two) - 1)


def _equally_sparse_match(s1, s2):
	if len(s1) > len(s2):
		return None
	matrix = np.zeros((len(s1), len(s2)))  # prepare matrix for results
	for i in range(len(s1)):
		for j in range(len(s2)):
			if s1[i] == s2[j]:
				if i == 0 or j == 0:  # if matched symbols are at the start of the sequence
					matrix[i][j] += 1  # if symbols matched - add 1
				else:
					matrix[i][j] = 1  # if symbols matched - add 1
	s2_indi = (np.vstack(
		[np.arange(matrix.shape[1]) + 1 for x in range(matrix.shape[0])])) * matrix  # convert matched 1's into indices
	s1_indi = (np.hstack(
		[np.expand_dims(np.arange(matrix.shape[0]), axis=1) + 1 for x in range(matrix.shape[1])])) * matrix
	s2diags = get_diags(s2_indi)  # get all diagonals
	s1diags = get_diags(s1_indi)
	if sum([sum(x) for x in s2diags]) == 0:
		return None
	nonzero_s2 = [[y - 1 for y in x if y != 0] for x in s2diags if sum(x) > 0]  # filter out empty lists
	nonzero_s1 = [[len(s1) - y + 1 for y in x if y != 0] for x in s1diags if sum(x) > 0]  # filter out empty lists
	# nonzero_s2 = [x for x in nonzero_s2 if len(x) >= 2]
	# nonzero_s1 = [x for x in nonzero_s1 if len(x) >= 2]
	matches = []
	for x, y in zip(nonzero_s1, nonzero_s2):
		if y[-1] + x[-1] < len(s2):
			matched_pattern = np.zeros((len(s1)+len(s2))//2)-1
			for z, w in zip(y, x):
				matched_pattern[int(-w)] = s2[int(z)]
			# matched_pattern = [(int(-w), s2[int(z)]) for z, w in zip(y, x)]
			next_symbol = s2[int(y[-1] + x[-1])]
			matches.append((matched_pattern, next_symbol))
	return matches


def fano_inequality(distinct_locations, entropy):
	"""
	Implementation of the Fano's inequality. Algorithm solves it and returns the solution.
	:param distinct_locations:
	:param entropy:
	:return:
	"""
	func = lambda x: (-(x * np.log2(x) + (1 - x) * np.log2(1 - x)) + (1 - x) * np.log2(
		distinct_locations - 1)) - entropy
	return fsolve(func, .9999)[0]


def to_labels(trajectories_frame):
	"""
	Adds labels column based on repeating geometries or coordinates
	:param trajectories_frame: TrajectoriesFrame object class
	:return: TrajectoriesFrame with labels column
	"""
	to_tranformation = trajectories_frame[trajectories_frame.geometry.is_valid]
	try:
		to_tranformation['labels'] = to_tranformation[to_tranformation._geom_cols[0]].astype(str) + ',' + \
		                               to_tranformation[to_tranformation._geom_cols[1]].astype(str)
		trajectories_frame['labels'] = trajectories_frame[to_tranformation._geom_cols[0]].astype(str) + ',' + \
		                             trajectories_frame[to_tranformation._geom_cols[1]].astype(str)
	except:
		to_tranformation['labels'] = to_tranformation.lat.astype(str) + ',' + to_tranformation.lon.astype(str)
		trajectories_frame['labels'] = trajectories_frame.lat.astype(str) + ',' + trajectories_frame.lon.astype(str)
	unique_coors = pd.DataFrame(pd.unique(to_tranformation['labels']))
	unique_coors.index = unique_coors.loc[:, 0]
	unique_coors.loc[:, 0] = range(len(unique_coors))
	sub_dict = unique_coors.to_dict()[0]
	trajectories_frame.astype({'labels': str})
	trajectories_frame['labels'] = trajectories_frame['labels'].map(sub_dict)
	return trajectories_frame
