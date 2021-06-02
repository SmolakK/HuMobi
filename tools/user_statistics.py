import sys
sys.path.append("..")
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


def fraction_of_empty_records(trajectories_frame, time_unit):
	"""
	Calculates the fraction q of empty records per user in a TrajectoriesFrame at given temporal resolution
	:param trajectories_frame: TrajectoriesFrame class object
	:param time_unit: time unit at which the fraction of empty records will be determined
	:return: A pandas Series with fractions of empty record for each user
	"""
	resampled = trajectories_frame.groupby(level=0).resample(time_unit, level=1).count().iloc[:,0]
	fractions = resampled.groupby(level=0).progress_apply(lambda x: (x[x==0]).count()/(x.count()))
	return fractions


def count_records(trajectories_frame):
	"""
	Returns total number of records for each user
	:param trajectories_frame: TrajectoriesFrame class object
	:return: A pandas Series with a count of records for each user
	"""
	counts = trajectories_frame.groupby(level=0).agg('count').iloc[:, 0]
	return counts


def count_records_per_time_frame(trajectories_frame, resolution):
	"""
	Returns total number of records for each user per time frame
	:param trajectories_frame: TrajectoriesFrame class object
	:param resolution: a time frame per which the count will be calculated
	:return: A pandas Series with a count of records for each user
	"""
	level_values = trajectories_frame.index.get_level_values
	time_frame_counts = trajectories_frame.groupby([level_values(0)] + [pd.Grouper(freq=resolution, level=-1)]).agg('count').iloc[:, 0]
	return time_frame_counts


def user_trajectories_duration(trajectories_frame, time_unit, count_empty=True):
	"""
	Returns the total duration of users' trajectories
	:param trajectories_frame: TrajectoriesFrame class object
	:param time_unit: time unit in which duration will be expressed
	:param count_empty: if empty records should be included
	:return: a pandas Series with the duration of each user's trajectory
	"""
	total_time_duration = trajectories_frame.groupby(level=0).resample(time_unit, level=1).count().iloc[:, 0]
	if count_empty:
		total_time_duration = total_time_duration.groupby(level=0).count()
	else:
		total_time_duration[total_time_duration == 1].groupby(level=0).count()
	return total_time_duration


def consecutive_record(trajectories_frame, time_unit):
	"""
	Calculates the maximum length of consecutive records for each user
	:param trajectories_frame: TrajectoriesFrame object class
	:param time_unit: time unit at which the consecutive records will be counded
	:return: a pandas Series with the maximum length of consecutive records for each user
	"""
	total_time_duration = trajectories_frame.groupby(level=0).apply(lambda x: x.groupby(pd.Grouper(freq=time_unit, level=1)).count()).iloc[:,0]
	total_time_duration[total_time_duration > 0] = 1
	total_time_duration = total_time_duration.groupby(level=0).progress_apply(
		lambda x: x * x.groupby((total_time_duration != total_time_duration.shift()).cumsum()).cumcount() + 1)
	return total_time_duration


def get_max(statistical_series):
	"""
	Get maximum value for each group
	:param statistical_series: Multiindex series
	:return: A series with maximum value for each group
	"""
	return statistical_series.groupby(level=0).agg('max')


def get_min(statistical_series):
	"""
	Get minimum value for each group
	:param statistical_series: Multiindex series
	:return: A series with minimum value for each group
	"""
	return statistical_series.groupby(level=0).agg('min')


def get_mean(statistical_series):
	"""
	Get mean value for each group
	:param statistical_series: Multiindex series
	:return: A series with mean value for each group
	"""
	return statistical_series.groupby(level=0).agg('mean')
