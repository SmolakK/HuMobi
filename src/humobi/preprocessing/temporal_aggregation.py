import pandas as pd
from humobi.structures.trajectory import TrajectoriesFrame
from humobi.preprocessing.filters import fill_gaps
import concurrent.futures as cf
from tqdm import tqdm


class TemporalAggregator():
	"""
	A class for temporal aggregation of TrajectoriesFrame. It aggregates each users' trajectory separately and returns
	single TrajectoriesFrame.
	"""

	def __init__(self, resolution):
		"""
		Class initialisation

		Args:
			resolution: resolution to which data will be aggregated
		"""
		self._resolution = resolution

	@property
	def resolution(self):
		return self._resolution

	def _user_aggregate(self, uid, single_user):
		"""
		Single trajectory aggregation.  For each time frame the longest visited location is selected.

		Args:
			uid: user's identifier
			single_user: user's trajectory

		Returns:
			An aggregated trajectory
		"""
		single_user['temp'] = range(len(single_user))
		df_time = (single_user[['temp', 'start', 'end']].set_index('temp').stack().reset_index(level=-1, drop=True).rename('time').to_frame())
		df_time['time'] = pd.to_datetime(df_time['time'])
		df_time = (df_time.groupby('temp').apply(lambda x: x.set_index('time').resample(self.resolution).asfreq()).reset_index())
		df_time = df_time.merge(single_user[single_user.columns.values])
		df_time['start'] = pd.to_datetime(df_time['start'])
		df_time['end'] = pd.to_datetime(df_time['end'])
		df_time = df_time.set_index('time')
		grouped = df_time.resample(self.resolution)
		for_concat = []
		tdelta = pd.Timedelta(self.resolution)
		for time_bin, data_bin in grouped:
			if len(data_bin) > 1:
				start = data_bin.index.min()
				end = start + tdelta
				longest_stay = data_bin.groupby('temp').apply(
					lambda x: x['end'] - x['start'] if ((x['end'] < end).all() and (x['start'] > start).all()).all() else end - x[
						'start'] if (x['end'] > end).all() else x['start'] - start).idxmax()[0]
				selected = data_bin[data_bin['temp'] == longest_stay]
			else:
				selected = data_bin
			for_concat.append(selected)
		df_time = pd.concat(for_concat)
		df_time = pd.concat({uid: df_time})
		return df_time

	def aggregate(self, trajectories_frame, fill_method = None, drop_empty = False, parallel = True):
		"""
		Aggregates TrajectoriesFrame temporally to given temporal resolution.
		For each time frame the longest visited location is selected.

		Args:
			trajectories_frame: TrajectoriesFrame class object
			fill_method: Empty records filling method, if 'None' then data is not filled in. Any other value causes use of forward fill.
			drop_empty (default = False): Determines if empty records should be removed.
			parallel (default = True): Determines if parallel computing should be used.

		Returns:
			A temporally aggregated TrajectoriesFrame object
		"""
		if not hasattr(trajectories_frame, '_geom_cols'):
			trajectories_frame = TrajectoriesFrame(trajectories_frame)
		trajectories_frame_grouped = trajectories_frame.groupby(level=0)
		for_concat = []
		if parallel:
			with cf.ThreadPoolExecutor() as executor:
				indis = [indi for indi, val in trajectories_frame_grouped]
				vals = [val for indi, val in trajectories_frame_grouped]
				results = list(tqdm(executor.map(self._user_aggregate, indis, vals), total=len(trajectories_frame_grouped)))
			for result in results:
				for_concat.append(result)
		else:
			for uid, user_data in trajectories_frame_grouped:
				if not (user_data['lat'].isna()).all():
					single_user_aggregated = self._user_aggregate(uid, user_data)
					for_concat.append(single_user_aggregated)
		trajectories_frame = pd.concat(for_concat).groupby(level=0).apply(lambda x: x.groupby(pd.Grouper(level=1,freq=self.resolution)).agg('first'))
		if fill_method is not None:
			trajectories_frame = fill_gaps(trajectories_frame, fill_method)
			trajectories_frame = fill_gaps(trajectories_frame,'bfill')
		trajectories_frame = trajectories_frame.drop('geometry', axis=1)
		if drop_empty:
			trajectories_frame = trajectories_frame[~trajectories_frame['lat'].isna()]

		trajectories_frame = TrajectoriesFrame(trajectories_frame)
		return trajectories_frame
