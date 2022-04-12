import pandas as pd
import numpy as np


def top_time(ind=None, gs=None):
	"""
	Selects the location (by coordinates) which was visited for the longest period during given time interval

	Args:
		ind: user id
		gs: GeoDataFrame from groupby execution containing all the data in the given time interval

	Returns:
		user id (if given) and the data for the longest visited location
	"""
	aggregated = []
	for tstamp, g in gs:  # for each record in the GeoDataFrame
		if len(g) > 1:  # if there is more than one record
			diff_places = (g['geometry'].shift(-1) != g['geometry']).iloc[:-1]  # checks when coordinates change
			if diff_places.any():  # if there is change in locations
				g_res = g.reset_index()  # drop index
				diffs = g_res.shift(-1)['datetime'] - g_res['datetime']  # find time differences (spent in location)
				joined_dfs = g_res.join(diffs, rsuffix='a')  # add them to locations
				joined_dfs['geometry'] = g_res['geometry'].astype(str)  # copy geometry as string
				point_max = joined_dfs.groupby('geometry')['datetimea'].sum().idxmax()  # grouping locations find the longest time sum
				selected = g[g['geometry'].astype(str) == point_max]  # select the location with the highest total time
			else:
				selected = g  # if one location visited - copy GeoDataFrame
		else:
			selected = g
		aggregated.append(selected)
	if ind is None:
		return pd.concat(aggregated)
	else:
		return ind, pd.concat(aggregated)


def mode_geoseries(ind, gs):
	"""
	Calculates mode for GeoSeries

	Args:
		ind: identifier
		gs: GeoSeries

	Returns:
		identifier and a mode for GeoSeries
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


def rowwise_average(gs, row_count=None):
	"""
	Calculates an average for each row in each group - rowwise.

	Args:
		gs: GeoSeries
		row_count: defines how much rows should be considered

	Returns:
		averaged GeoSeries rowwise
	"""
	if row_count is None:
		row_count = gs.groupby(level=0).size().max()
	return pd.Series([gs.groupby(level=0).nth(n).mean() for n in range(row_count)])


def groupwise_average(gs):
	"""
	Calculates an average from each group of GeoSeries

	Args:
		gs: GeoSeries

	Returns:
		averaged GeoSeries
	"""
	return gs.groupby(level=0).mean()


def groupwise_normalise(gs):
	"""
	Normalises each group of GeoSeries

	Args:
		gs: GeoSeries

	Returns:
		normalised GeoSeries
	"""
	return gs.groupby(level=0).apply(lambda x: x / x.sum())


def groupwise_expansion(gs):
	"""
	Calculates expanding mean for each group of GeoSeries

	Args:
		gs: GeoSeries

	Returns:
		averaged GeoSeries
	"""
	return gs.groupby(level=0).expanding().mean()


def convert_to_distribution(gs, bin_size=None, num_of_classes=None):
	"""
	Converts statistics from Series into a distribution

	Args:
		gs: GeoSeries with a statistic
		bin_size: size of a bin for histogram (if used, num_of_classes cannot be determined)
		num_of_classes: number of groups in histogram (if used, bin_size cannot be determined)

	Returns:
		A Series with distribution of desired parameters. Bins are indicated as index.
	"""
	lower = gs.min()
	upper = gs.max()
	if num_of_classes is None:
		gs = gs.groupby(pd.cut(gs, np.arange(lower, upper, bin_size))).count()
	if bin_size is None:
		gs = gs.groupby(pd.cut(gs, np.linspace(lower, upper, num_of_classes+1))).count()
	return gs/gs.sum()


def normalize(gs, name):
	"""
	Normalises a column of geoseries.

	Args:
		gs: GeoSeries
		name: Column to normalise

	Retruns:
		GeoSeries with given column normalised
	"""
	gs_column = gs.loc[:, name]
	gs_column = gs_column / gs_column.sum()
	if gs_column.sum() != 1:
		offset = 1 - gs_column.sum()
		max_ind = gs_column.idxmax()
		gs_column.loc[max_ind] += offset
	gs.loc[:, name] = gs_column
	return gs


def total_normalise(gs):
	"""
	Performs complete normalisation of GeoSeries

	Args:
		gs: GeoSeries

	Returns:
		normalised GeoSeries
	"""
	return gs / gs.sum()


def start_end(trajectories_frame):
	"""
	Compresses stops in TrajectoriesFrame by adding start and end of visits in locations

	Args:
		trajectories_frame: TrajectoriesFrame object class

	Returns:
		compressed TrajectoriesFrame
	"""
	to_concat = []
	if 'date' not in trajectories_frame.columns:
		trajectories_frame['date'] = trajectories_frame.index.get_level_values(1)
	for gs in trajectories_frame.groupby(level=0):
		firsts = gs[1][gs[1]['geometry'].shift() != gs[1]['geometry']]
		lasts = gs[1][gs[1]['geometry'].shift(-1) != gs[1]['geometry']]
		firsts.loc[:, 'start'] = firsts['date']
		lasts = lasts.set_index(firsts.index)
		firsts.loc[:, 'end'] = lasts['date']
		firsts = firsts[firsts['start'] != firsts['end']]
		to_concat.append(firsts)
	return pd.concat(to_concat)
