import pandas as pd
import numpy as np


def mode_geoseries(ind,gs):
	"""
	Calculates mode for GeoSeries
	:param ind: identifier
	:param gs: GeoSeries
	:return: identifier and a mode for GeoSeries
	"""
	aggregated = []
	for g in gs:
		if g[1].empty:
			aggregated.append(None)
		else:
			selected = g[1].mode()
			selected = selected.set_index(g[1].index)
			aggregated.append(selected)
	return ind, pd.concat(aggregated)


def rowwise_average(gs, row_count = None):
	"""
	Calculates an average for each row in each group - rowwise.
	:param gs: GeoSeries
	:param row_count: defines how much rows should be considered
	:return: averaged GeoSeries rowwise
	"""
	if row_count is None:
		row_count = gs.groupby(level=0).size().max()
	return pd.Series([gs.groupby(level=0).nth(n).mean() for n in range(row_count)])


def groupwise_average(gs):
	"""
	Calculates an average from each group of GeoSeries
	:param gs: GeoSeries
	:return: averaged GeoSeries
	"""
	return gs.groupby(level=0).mean()


def groupwise_normalise(gs):
	"""
	Normalises each group of GeoSeries
	:param gs: GeoSeries
	:return: normalised GeoSeries
	"""
	return gs.groupby(level=0).apply(lambda x: x / x.sum())


def groupwise_expansion(gs):
	"""
	Calculates expanding mean for each group of GeoSeries
	:param gs: GeoSeries
	:return: averaged GeoSeries
	"""
	return gs.groupby(level=0).expanding().mean()


def total_normalise(gs):
	"""
	Performs complete normalisation of GeoSeries
	:param gs: GeoSeries
	:return: normalised GeoSeries
	"""
	return gs/gs.sum()


def start_end(trajectories_frame):
	"""
	Compresses stops in TrajectoriesFrame by adding start and end of visits in locations
	:param trajectories_frame: TrajectoriesFrame object class
	:return: compressed TrajectoriesFrame
	"""
	to_concat = []
	if 'date' not in trajectories_frame.columns:
		trajectories_frame['date'] = trajectories_frame.index.get_level_values(1)
	for gs in trajectories_frame.groupby(level=0):
		firsts = gs[1][gs[1]['geometry'].shift() != gs[1]['geometry']]
		lasts = gs[1][gs[1]['geometry'].shift(-1) != gs[1]['geometry']]
		firsts.loc[:,'start'] = firsts['date']
		lasts = lasts.set_index(firsts.index)
		firsts.loc[:,'end'] = lasts['date']
		firsts = firsts[firsts['start'] != firsts['end']]
		to_concat.append(firsts)
	return pd.concat(to_concat)
