# IMPORTS
from src.humobi.preprocessing.temporal_aggregation import TemporalAggregator
import numpy as np
import pandas as pd
from src.humobi.tools.processing import groupwise_normalise, groupwise_expansion
from src.humobi.misc.utils import *
from src.humobi.misc.utils import _repeatfinder_dense, _repeatfinder_sparse, _repeatfinder_equally_sparse, \
	_global_align, _iterative_global_align
from tqdm import tqdm
from src.humobi.structures.trajectory import TrajectoriesFrame
tqdm.pandas()
import concurrent.futures as cf
from math import ceil
from random import sample
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def num_of_distinct_locations(trajectories_frame):
	"""
	Returns a number of distinct location in the trajectory. First looks for 'labels' column.

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		a Series with the number of unique locations for each user
	"""
	if isinstance(trajectories_frame, pd.DataFrame):
		return trajectories_frame.groupby(level=0).progress_apply(lambda x: len(pd.unique(x['labels'])))
	else:
		return trajectories_frame.groupby(level=0).progress_apply(lambda x: pd.unique(x).shape[0])


def visitation_frequency(trajectories_frame):
	"""
	Calculates visitiation frequency for each user in the TrajectoriesFrame.

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		a Series with the visitation frequency for each user and for each location. Sorted descendingly.
	"""
	lat_col = trajectories_frame._geom_cols[0]
	lon_col = trajectories_frame._geom_cols[1]
	frequencies = trajectories_frame.groupby(level=0).progress_apply(
		lambda x: x.groupby([lat_col, lon_col]).count()).iloc[:, 0]
	frequencies = frequencies.groupby(level=0).progress_apply(lambda x: x.sort_values(ascending=False))
	frequencies = groupwise_normalise(frequencies)
	return frequencies


def _filter_distinct_locations(trajectories_frame):
	to_concat = []
	for ind, vals in tqdm(trajectories_frame.groupby(level=0), total=len(trajectories_frame.groupby(level=0)) ):
		if len(vals) == 1:
			to_concat.append(uniq)
			continue
		else:
			uniq = (vals['geometry'][vals['geometry'].is_valid].drop_duplicates())
			to_concat.append(uniq)
	return pd.concat(to_concat)


def distinct_locations_over_time(trajectories_frame, resolution='1H', reaggregate=False):
	"""
	Calculates the number of distinct location visited in the movement trajectory over time.

	Args:
		trajectories_frame: TrajectoriesFrame class object
		resolution: determines the time unit for the metric
		reaggregate: if true, data are first reagregated to given time unit

	Returns:
		a Series with the number of unique locations visited up to each time step in the movement trajectory
	"""
	if reaggregate:
		temp_agg = TemporalAggregator(resolution)
		trajectories_frame = temp_agg.aggregate(trajectories_frame)
	trajectories_frame = _filter_distinct_locations(trajectories_frame)
	distinct_locations = trajectories_frame.groupby(level=0).resample(resolution, level=1).count()
	distinct_locations = distinct_locations.groupby(level=0).cumsum()
	return distinct_locations


def jump_lengths(trajectories_frame):
	"""
	Calculates jump lengths between each step in the trajectory

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		a Series with jump lengths between consecutive records
	"""
	jumps = trajectories_frame.dropna().geometry.groupby(level=0).progress_apply(lambda x: x.distance(x.shift()))
	return jumps


def nonzero_trips(trajectories_frame):
	"""
	Counts all trips that had distance larger than 0.

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		a Series with a count of nonzero trips for each user
	"""
	jumps = jump_lengths(trajectories_frame).dropna().droplevel(1)
	return jumps[jumps != 0].groupby(level=0).count()


def self_transitions(trajectories_frame):
	"""
	Calculates the number of self transitions for each user

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Retruns
		a Series with the number of self transitions for each user
	"""
	if isinstance(trajectories_frame, pd.Series):
		self_transitions_mask = (trajectories_frame == trajectories_frame.shift())
	else:
		if not hasattr(trajectories_frame, '_geom_cols'):
			trajectories_frame = TrajectoriesFrame(trajectories_frame)
		coordinates_frame = trajectories_frame[[trajectories_frame._geom_cols[0], trajectories_frame._geom_cols[1]]]
		self_transitions_mask = (coordinates_frame == coordinates_frame.shift()).all(axis=1)
	empty_mask = (~self_transitions_mask).groupby(level=0).progress_apply(lambda x: x.all())
	empty_mask = empty_mask[empty_mask == True].index
	self_transitions_only = trajectories_frame[self_transitions_mask]
	empty_self_transitions = pd.DataFrame([0 for x in range(len(empty_mask))], index=empty_mask)
	if isinstance(trajectories_frame, pd.Series):
		self_transitions_only = self_transitions_only.groupby(level=0).count()
	else:
		self_transitions_only = self_transitions_only.groupby(level=0).count()[self_transitions_only.columns[0]]
	if len(empty_self_transitions) > 0:
		self_transitions_only.append(empty_self_transitions.iloc[:, 0]).sort_index()
	return self_transitions_only


def waiting_times(trajectories_frame, time_unit='h'):
	"""
	Calculates waiting times for each transition in TrajectoriesFrame

	Args:
		trajectories_frame: TrajectoriesFrame class object
		time_unit: time unit in which waiting times will be expressed

	Returns:
		A series with waiting times for each transition for each user
	"""
	trajectories_frame = trajectories_frame.dropna()
	transitions_only = trajectories_frame[
		trajectories_frame.geometry.groupby(level=0).progress_apply(lambda x: x.shift(-1) != x)]
	transitions_only['dt'] = transitions_only.index.get_level_values(1)
	times = transitions_only.groupby(level=0).dt.progress_apply(lambda x: (x - x.shift()).astype('timedelta64[%s]' % time_unit))
	return times


def center_of_mass(trajectories_frame):
	"""
	Calculates a center of mass for each user's trajectory

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		a GeoSeries with centers of mass of each user's trajectory
	"""
	return trajectories_frame.dissolve(by=trajectories_frame.index.get_level_values(0)).centroid


def radius_of_gyration(trajectories_frame, time_evolution=True):
	"""
	Calculates radii of gyration for each user. Optionally uses time steps to express their growth.

	Args:
		trajectories_frame: TrajectoriesFrame class object
		time_evolution: If true, radii of gyration are calculated over time

	Returns:
		a Series with radii of gyration for each user
	"""
	mean_locs = center_of_mass(trajectories_frame)
	to_concat_dict = {}
	to_concat_list = []
	for ind, vals in tqdm(trajectories_frame.groupby(level=0), total=len(trajectories_frame.groupby(level=0))):
		vals = vals.dropna()
		rog_ind = vals.distance(mean_locs.loc[ind]) ** 2
		if time_evolution:
			rog_ind = groupwise_expansion(np.sqrt(rog_ind))
			to_concat_list.append(rog_ind)
		else:
			rog_ind = np.sqrt(rog_ind.mean())
			to_concat_dict[ind] = rog_ind
	if time_evolution:
		radius = pd.concat(to_concat_list)
	else:
		radius = pd.DataFrame.from_dict(to_concat_dict, orient='index')
	return radius


def mean_square_displacement(trajectories_frame, from_center=False, time_evolution=True, reference_locs=None):
	"""
	Calculates mean square displacements for each user. Optionally uses time steps to express their growth.

	Args:
		trajectories_frame: TrajectoriesFrame class object
		from_center: If true, displacement is calculated from the trajectory ceneter, if false - from the first point
		time_evolution: If true, mean square displacements are calculated over time
		reference_locs: allows to give reference locations for each trajectory explicitly

	Returns:
		a Series with mean square displacements for each user
	"""
	to_concat_dict = {}
	to_concat_list = []
	if reference_locs is None:
		if from_center:
			reference_locs = center_of_mass(trajectories_frame)
		else:
			reference_locs = trajectories_frame.groupby(level=0).head(1).droplevel(1).geometry
	for ind, vals in tqdm(trajectories_frame.groupby(level=0), total=len(trajectories_frame.groupby(level=0))):
		vals = vals.dropna()
		msd_ind = (vals.distance(reference_locs.loc[ind]) ** 2)
		if time_evolution:
			msd_ind = groupwise_expansion(msd_ind)
			to_concat_list.append(msd_ind)
		else:
			msd_ind = msd_ind.mean()
			to_concat_dict[ind] = msd_ind
	if time_evolution:
		msd = pd.concat(to_concat_list)
	else:
		msd = pd.DataFrame.from_dict(to_concat_dict, orient='index')
	return msd


def return_time(trajectories_frame, time_unit='h', by_place=False):
	"""
	Calculates return times for each unique location in each user's trajectory.

	Args:
		trajectories_frame: TrajectoriesFrame class object
		time_unit: time unit in which return times will be expressed
		by_place: If true, return times are expressed for each place globally

	Returns:
		a Series with return times
	"""
	if not hasattr(trajectories_frame, '_geom_cols'):
		trajectories_frame = TrajectoriesFrame(trajectories_frame)
	lat_col = trajectories_frame[trajectories_frame._geom_cols[0]]
	lon_col = trajectories_frame[trajectories_frame._geom_cols[1]]
	trajectories_frame['datetime_temp'] = trajectories_frame.index.get_level_values(1)
	to_concat = []
	for ind, vals in tqdm(trajectories_frame.groupby(level=0), total=len(trajectories_frame.groupby(level=0))):
		concat_level = {}
		for place, vals2 in vals.groupby([lat_col, lon_col]):
			shifts = (vals2.datetime_temp - vals2.datetime_temp.shift()).astype('timedelta64[%s]' % time_unit)
			concat_level[place] = shifts
		to_concat.append(pd.concat(concat_level))
	return_times = pd.concat(to_concat)
	if by_place:
		return_times = return_times.groupby(level=2).progress_apply(
			lambda x: x.groupby(level=[0, 1]).agg(['count', 'mean']).dropna())
		return_times = return_times.groupby(level=0).progress_apply(lambda x: x.sort_values('count', ascending=False))
	else:
		return_times = return_times.groupby(level=2).progress_apply(lambda x: x.sort_values(ascending=False)).droplevel(
			[1, 2])
	return return_times


def random_entropy(trajectories_frame):
	"""
	Calculates random entropy for each user in TrajectoriesFrame

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		a Series with random entropies for each user
	"""
	return trajectories_frame.groupby(level=0).progress_apply(lambda x: np.log2(len(pd.unique(x.geometry))))


def unc_entropy(trajectories_frame):
	"""
	Calculates uncorrelated entropy for each user in TrajectoriesFrame

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		a Series with uncorrelated entropies for each user
	"""
	frequencies = visitation_frequency(trajectories_frame)
	return frequencies.groupby(level=0).progress_apply(lambda x: -np.sum([pk * np.log2(pk) for pk in x if pk != 0]))


def _fit_func(x, a, b, c):
	return a * np.exp(b * x) + c


def _real_entropy(indi, gs):
	"""
	Calculates actual entropy for series of symbols

	Args:
		indi: unique identifier
		gs: series of symbols

	Returns:
		a unique identifier and entropy value
	"""
	return indi, np.power(np.mean(matchfinder(gs)), -1) * np.log2(len(gs))


def _real_scalling_entropy(indi, trace, estimation_method='unc'):
	"""
	Calculates actual entropy for trajectories. If trajectory has missing data, uses estimation. Uncorrelated entropy-based
	estimation is used.

	Args:
		indi: unique identifier
		trace: movement trajectory

	Returns:
		a unique identifier and actual entropy
	"""
	empty_fraction = trace.isnull().sum() / trace.shape[0]
	if empty_fraction < .15:
		return _real_entropy(indi, trace)
	elif empty_fraction >.9:
		raise ValueError("Fraction of missing records (q = %f) to high to estimate real entropy!" % empty_fraction)
	estimation_step = ceil((.9 - empty_fraction) / .05)
	range_to_empty = [empty_fraction + .05 * x for x in range(estimation_step)]
	scaling_features = []
	uncs = []
	real_qs = []
	if estimation_method == 'unc':
		visit_freq = trace.value_counts() / trace.shape[0]  # FOR UNCORRELATED ENTROPY-BASED ESTIMATION
		Sunc_baseline = -np.sum(visit_freq * np.log2(visit_freq))  # FOR UNCORRELATED ENTROPY-BASED ESTIMATION
	elif estimation_method == 'shuff':
		Sunc_baseline = np.power(np.mean(matchfinder(trace.sample(frac=1).reset_index(drop=True))), -1) * \
		                np.log2(len(
			                trace.sample(frac=1).reset_index(drop=True)))  # FOR SHUFFLED TRAJECTORY-BASED ESTIMATION
	for q in range_to_empty[1:]:
		trace_copy2 = trace.copy()
		points_to_remove = sample(set(trace_copy2[~trace_copy2.isnull()].index),
		                          int(round((q - empty_fraction) * len(trace_copy2))))
		trace_copy2.loc[points_to_remove] = None
		Strue = np.power(np.mean(matchfinder(trace_copy2)), -1) * np.log2(len(trace_copy2))
		trace_shuffled = trace_copy2.sample(frac=1).reset_index(drop=True)
		if estimation_method == 'unc':
			visit_freq = trace_copy2.value_counts() / trace_copy2.shape[0]  # FOR UNCORRELATED ENTROPY-BASED ESTIMATION
			Sunc = -np.sum(visit_freq * np.log2(visit_freq))  # FOR UNCORRELATED ENTROPY-BASED ESTIMATION
		elif estimation_method == 'shuff':
			Sunc = np.power(np.mean(matchfinder(trace_shuffled)), -1) * np.log2(len(trace_shuffled)) # FOR SHUFFLED TRAJECTORY-BASED ESTIMATION
		scaling_features.append(np.log2(Strue / Sunc))
		uncs.append(Sunc)
		real_qs.append(sum(trace_copy2.isnull()) / len(trace_copy2))
	try:
		popt, pcov = curve_fit(_fit_func, real_qs, scaling_features, maxfev=12000, p0=[0.1, 2, 0.1])
		if sum(scaling_features) == 0 and r2_score(scaling_features, [_fit_func(x, *popt) for x in real_qs]) < .9:
			a, b = np.polyfit(real_qs, scaling_features, 1)
			return indi, np.power(2, b) * Sunc_baseline
		else:
			return indi, np.power(2, _fit_func(0, *popt)) * Sunc_baseline
	except:
		a, b = np.polyfit(real_qs, scaling_features, 1)
		return indi, np.power(2, b) * Sunc_baseline


def real_entropy(trajectories_frame):
	"""
	Calculates an actual entropy for each user in TrajectoriesFrame

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		a Series with actual entropies for each user
	"""
	result_dic = {}
	with cf.ThreadPoolExecutor() as executor:
		try:
			args = [val.labels for indi, val in trajectories_frame.groupby(level=0)]
		except KeyError:
			args = [val for indi, val in trajectories_frame.groupby(level=0)]
		ids = [indi for indi, val in trajectories_frame.groupby(level=0)]
		results = list(tqdm(executor.map(_real_scalling_entropy, ids, args), total=len(ids)))
	for result in results:
		result_dic[result[0]] = result[1]
	return pd.Series(np.fromiter(result_dic.values(), dtype=float), index=np.fromiter(result_dic.keys(), dtype=int))


def random_predictability(trajectories_frame):
	"""
	Calculates random entropy and predictability.

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		a Series with random entropy and predictability for each user
	"""
	distinct_locations = num_of_distinct_locations(trajectories_frame)
	rand_ent = random_entropy(trajectories_frame)
	merged = pd.DataFrame([distinct_locations, rand_ent], index=['locations', 'entropy'])
	return merged.progress_apply(lambda x: fano_inequality(x['locations'], x['entropy'])), rand_ent


def unc_predictability(trajectories_frame):
	"""
	Calculates uncorrelated entropy and predictability.

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		a Series with uncorrelated entropy and predictability for each user
	"""
	distinct_locations = num_of_distinct_locations(trajectories_frame)
	unc_ent = unc_entropy(trajectories_frame)
	merged = pd.DataFrame([distinct_locations, unc_ent], index=['locations', 'entropy'])
	return merged.progress_apply(lambda x: fano_inequality(x['locations'], x['entropy'])), unc_ent


def real_predictability(trajectories_frame):
	"""
	Calculates actual entropy and predictability.

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		a Series with actual entropy and predictability for each user
	"""
	distinct_locations = num_of_distinct_locations(trajectories_frame)
	real_ent = real_entropy(trajectories_frame)
	merged = pd.DataFrame([distinct_locations, real_ent], index=['locations', 'entropy'])
	return merged.progress_apply(lambda x: fano_inequality(x['locations'], x['entropy'])), real_ent


def stationarity(trajectories_frame):
	"""
	Calculates the stationarity according to Teixeira et al. (2019) as the average stay length in the location.

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		a Series with stationarity values for each user
	"""
	trajectories_frame = trajectories_frame.dropna()
	stationarity_frame = trajectories_frame.labels.groupby(level=0).apply(
		lambda x: x.groupby((x != x.shift()).cumsum()).size()-1)
	stationarity_frame = stationarity_frame.groupby(level=0).sum()
	size_frame = trajectories_frame.groupby(level=0).apply(lambda x: x.size)
	return stationarity_frame/size_frame


def regularity(trajectories_frame):
	"""
	Calculates the regularity according to Teixeira et al. (2019) as the ratio of sequence lenght and the number of
	unique symbols

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		a Series with regularity values for each user
	"""
	regul = trajectories_frame.groupby(level=0).progress_apply(lambda x: len(x)) / num_of_distinct_locations(
		trajectories_frame)
	return regul


def _repeatability_dense(indi, S1, S2):
	S1 = S1.values
	S2 = S2.values
	return indi, _repeatfinder_dense(S1, S2)


def _repeatability_sparse(indi, S1, S2):
	S1 = S1.values
	S2 = S2.values
	return indi, _repeatfinder_sparse(S1, S2)


def _repeatability_equally_sparse(indi, S1, S2):
	S1 = S1.values
	S2 = S2.values
	return indi, _repeatfinder_equally_sparse(S1,S2)


def _global_alignment(indi, S1, S2):
	S1 = S1.values
	S2 = S2.values
	return indi, _global_align(S1, S2)


def _iterative_global_alignment(indi, S1, S2):
	S1 = S1.values
	S2 = S2.values
	return indi, _iterative_global_align(S1, S2)


def repeatability_sparse(train_frame, test_frame):
	"""
	Calculates the sparse repeatability metric proposed in the paper Smolak et al. (2022) as the number of
	transitions repeating in two sequences in the same order. Matched transitions can be separated by a gap of any size.
	Returns normalised values.

	Args:
		train_frame: TrajectoriesFrame class object with the training data
		test_frame:TrajectoriesFrame class object with the test data

	Returns:
		a Series with sparse repeatability values for each user
	"""
	result_dic = {}
	with cf.ThreadPoolExecutor() as executor:
		args_train = [val.labels for indi, val in train_frame.groupby(level=0)]
		args_test = [val.labels for indi, val in test_frame.groupby(level=0)]
		ids = [indi for indi, val in test_frame.groupby(level=0)]
		results = list(tqdm(executor.map(_repeatability_sparse, ids, args_train, args_test), total=len(ids)))
	for result in results:
		result_dic[result[0]] = result[1]
	return pd.Series(np.fromiter(result_dic.values(), dtype=float), index = np.fromiter(result_dic.keys(), dtype = int))


def repeatability_dense(train_frame, test_frame):
	"""
	Calculates the dense repeatability metric proposed in the paper Smolak et al. (2022) as the number of
	transitions repeating in two sequences in the same order. Matched transitions cannot be separated by a gap.
	Returns normalised values.

	Args:
		train_frame: TrajectoriesFrame class object with the training data
		test_frame:TrajectoriesFrame class object with the test data

	Returns:
		a Series with dense repeatability values for each user
	"""
	result_dic = {}
	with cf.ThreadPoolExecutor() as executor:
		args_train = [val.labels for indi, val in train_frame.groupby(level=0)]
		args_test = [val.labels for indi, val in test_frame.groupby(level=0)]
		ids = [indi for indi, val in test_frame.groupby(level=0)]
		results = list(tqdm(executor.map(_repeatability_dense, ids, args_train, args_test), total=len(ids)))
	for result in results:
		result_dic[result[0]] = result[1]
	return pd.Series(np.fromiter(result_dic.values(), dtype=float), index=np.fromiter(result_dic.keys(), dtype=int))


def repeatability_equally_sparse(train_frame, test_frame):
	"""
	Calculates the equally sparse repeatability metric proposed in the paper Smolak et al. (2022) as the number of
	transitions repeating in two sequences in the same order. Matched transitions can be separated by a gap of any size.
	Matched transitions HAVE TO BE spearated by the gap of the same size.
	Returns normalised values.

	Args:
		train_frame: TrajectoriesFrame class object with the training data
		test_frame:TrajectoriesFrame class object with the test data

	Returns:
		a Series with equally sparse repeatability values for each user
	"""
	result_dic = {}
	with cf.ThreadPoolExecutor() as executor:
		args_train = [val.labels for indi, val in train_frame.groupby(level=0)]
		args_test = [val.labels for indi, val in test_frame.groupby(level=0)]
		ids = [indi for indi, val in test_frame.groupby(level=0)]
		results = list(tqdm(executor.map(_repeatability_equally_sparse, ids, args_train, args_test), total=len(ids)))
	for result in results:
		result_dic[result[0]] = result[1]
	return pd.Series(np.fromiter(result_dic.values(), dtype=float), index = np.fromiter(result_dic.keys(), dtype = int))


def global_alignment(train_frame, test_frame):
	"""
	Calculates the global alignment metric proposed in the paper Smolak et al. (2022) through the application of the
	Needleman-Wunsch algorithm on sequences for pairwise matching.
	Returns normalised values.

	Args:
		train_frame: TrajectoriesFrame class object with the training data
		test_frame: TrajectoriesFrame class object with the test data

	Returns:
		a Series with global alginment metric values for each user
	"""
	result_dic = {}
	with cf.ThreadPoolExecutor() as executor:
		args_train = [val.labels for indi, val in train_frame.groupby(level=0)]
		args_test = [val.labels for indi, val in test_frame.groupby(level=0)]
		ids = [indi for indi, val in test_frame.groupby(level=0)]
		results = list(tqdm(executor.map(_global_alignment, ids, args_train, args_test), total=len(ids)))
	for result in results:
		result_dic[result[0]] = result[1]
	return pd.Series(np.fromiter(result_dic.values(), dtype=float), index = np.fromiter(result_dic.keys(), dtype = int))


def iterative_global_alignment(train_frame, test_frame):
	"""
	Calculates the iterative global alignment metric proposed in the paper Smolak et al. (2022) through the application 
	of the Needleman-Wunsch algorithm on sequences for pairwise matching with the iterative approach.
	Returns normalised values.

	Args:
		train_frame: TrajectoriesFrame class object with the training data
		test_frame: TrajectoriesFrame class object with the test data

	Returns:
		a Series with global alignment metric values for each user
	"""
	result_dic = {}
	with cf.ThreadPoolExecutor() as executor:
		args_train = [val.labels for indi, val in train_frame.groupby(level=0)]
		args_test = [val.labels for indi, val in test_frame.groupby(level=0)]
		ids = [indi for indi, val in test_frame.groupby(level=0)]
		results = list(tqdm(executor.map(_iterative_global_alignment, ids, args_train, args_test), total=len(ids)))
	for result in results:
		result_dic[result[0]] = result[1]
	return pd.Series(np.fromiter(result_dic.values(), dtype=float), index = np.fromiter(result_dic.keys(), dtype = int))
