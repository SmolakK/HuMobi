import numpy as np
import sys

sys.path.append("..")
import pandas as pd
from numba import cuda
from math import ceil


def normalize(gs, name):
	"""
	Normalises geoseries
	:param gs: GeoSeries
	:param name: Column to nomralise
	:return: GeoSeries with column normalised
	"""
	gs_column = gs.loc[:, name]
	gs_column = gs_column / gs_column.sum()
	if gs_column.sum() != 1:
		offset = 1 - gs_column.sum()
		max_ind = gs_column.idxmax()
		gs_column.loc[max_ind] += offset
	gs.loc[:, name] = gs_column
	return gs


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
	Implements fast moving acerage
	:param a: input data array
	:param n: window size
	:return: processed data array
	"""
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


def _matchfinder(start_idx, gs, data_len):
	"""
	Finds the shortest nor repeating sequences according to Lempel-Ziv algorithm
	:param start_idx: starting point in the aray from which search will be started
	:param gs: symbol series
	:param data_len: data length
	:return: current starting index and the shortes non-repeating subsequence length
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
	Finds the shortest nor repeating sequences according to Lempel-Ziv algorithm. Algorithm adaptation for GPU.
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
			output[pos] = end_distance + 1 #CHANGED with XU paper
		else:
			output[pos] = max_subsequence_matched + 1


def matchfinder(gs):
	"""
	Finds the shortest nor repeating sequences according to Lempel-Ziv algorithm
	:param gs: symbol series
	:return: the length of the shortes non-repeating subsequences at each step of sequence
	"""
	gs = gs.dropna()
	data_len = len(gs)
	gs = np.array(gs.values)
	output = np.zeros(data_len)
	output[0] = 1
	if True:
		threadsperblock = 256
		blockspergrid = ceil(data_len / threadsperblock)
		_matchfinder_gpu[threadsperblock, blockspergrid](gs, data_len, output)
	return output


def to_labels(trajectories_frame):
	"""
	Adds labels column based on repeating geometries or coordinates
	:param trajectories_frame: TrajectoriesFrame object class
	:return: TrajectoriesFrame with labels column
	"""
	try:
		trajectories_frame['labels'] = trajectories_frame[trajectories_frame._geom_cols[0]].astype(str) + ',' + \
		                               trajectories_frame[trajectories_frame._geom_cols[1]].astype(str)
	except:
		trajectories_frame['labels'] = trajectories_frame.lat.astype(str) + ',' + trajectories_frame.lon.astype(str)
	unique_coors = pd.DataFrame(pd.unique(trajectories_frame['labels']))
	unique_coors.index = unique_coors.loc[:, 0]
	unique_coors.loc[:, 0] = range(len(unique_coors))
	sub_dict = unique_coors.to_dict()[0]
	trajectories_frame.astype({'labels': str})
	trajectories_frame['labels'] = trajectories_frame['labels'].map(sub_dict)
	return trajectories_frame
