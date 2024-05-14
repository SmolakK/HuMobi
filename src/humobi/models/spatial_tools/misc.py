import pandas as pd
from humobi.misc.utils import to_labels


def normalize_array(arr):
	"""
	Normalizes given array (sum to 1)
	:param arr: Array to normalize
	:return: Normalized array
	"""
	arr = arr/arr.sum()
	if arr.sum() != 1:
		offset = 1 - arr.sum()
		arr[arr.argmax()] += offset
	return arr


def rank_freq(trajectories_frame, quantity = 2, nighttime_approach = True, day_start=9, day_end=17, night_start=2, night_end=5):
	"""
	Ranks locations visited by each user by the frequency of visits. Used to select the most important location
	:param nighttime_approach: Use an approach where the most visited locations at night are used to detect important locations
	:param trajectories_frame: TrajectoriesFrame class object
	:param quantity: how many locations should be returned
	:param day_start: the start of the daylight location
	:param day_end: the end of the daylight location
	:param night_start: the start of the nighttime location
	:param night_end: the end of the nighttime location
	:return: Ranked locations
	"""
	if not 'labels' in trajectories_frame:
		to_labels(trajectories_frame)
	filtered_trajectories_frame = trajectories_frame[trajectories_frame.geometry.is_valid]
	if nighttime_approach:
		filtered_trajectories_frame['hod'] = filtered_trajectories_frame.index.get_level_values(1).hour
		daylight_frame = filtered_trajectories_frame[
			(filtered_trajectories_frame.hod >= day_start) & (filtered_trajectories_frame.hod <= day_end)]
		if night_start > night_end:
			nighttime_frame = filtered_trajectories_frame[(filtered_trajectories_frame.hod >= night_start) | (
					filtered_trajectories_frame.hod <= night_end)]
		elif night_start < night_end:
			nighttime_frame = filtered_trajectories_frame[(filtered_trajectories_frame.hod >= night_start) & (
					filtered_trajectories_frame.hod <= night_end)]
		top_places_day = daylight_frame.groupby(
			[daylight_frame.index.get_level_values(0), 'labels']).count().reset_index()
		top_places_day = pd.DataFrame(top_places_day).groupby('user_id').apply(
			lambda x: x.sort_values('temp', ascending=False)).reset_index(drop=True)
		top_places_night = nighttime_frame.groupby(
			[nighttime_frame.index.get_level_values(0), 'labels']).count().reset_index()
		top_places_night = pd.DataFrame(top_places_night).groupby('user_id').apply(
			lambda x: x.sort_values('temp', ascending=False)).reset_index(drop=True)
		merged_day = pd.merge(top_places_day[['user_id', 'labels']], filtered_trajectories_frame.reset_index(),
		                      on=['user_id', 'labels'], how='left')[['user_id', 'labels', 'geometry']].drop_duplicates()
		merged_night = pd.merge(top_places_night[['user_id', 'labels']], filtered_trajectories_frame.reset_index(),
		                        on=['user_id', 'labels'], how='left')[
			['user_id', 'labels', 'geometry']].drop_duplicates()
		unstacked_work = merged_day.groupby('user_id')['geometry'].apply(
			lambda x: x.reset_index(drop=True)).unstack()
		unstacked_home = merged_night.groupby('user_id')['geometry'].apply(
			lambda x: x.reset_index(drop=True)).unstack()
	top_places = filtered_trajectories_frame.groupby(
		[filtered_trajectories_frame.index.get_level_values(0), 'labels']).count().reset_index()
	top_places = pd.DataFrame(top_places).groupby('user_id').apply(lambda x: x.sort_values('temp', ascending=False)).reset_index(drop=True)
	merged = pd.merge(top_places[['user_id', 'labels']], filtered_trajectories_frame.reset_index(), on=['user_id', 'labels'], how='left')[['user_id', 'labels', 'geometry']].drop_duplicates()
	unstacked = merged.groupby('user_id')['geometry'].apply(lambda x: x.reset_index(drop=True)).unstack()
	if quantity == 1 and nighttime_approach:
		out = unstacked_home.iloc[:,0]
	elif quantity >= 2 and nighttime_approach:
		unstacked_work.iloc[:, 0][unstacked_home.iloc[:, 0] == unstacked_work.iloc[:, 0]] = None
		unstacked_work_frame = {}
		for n in unstacked_work.index:
			work_collapsed = unstacked_work.loc[n,:][~unstacked_work.loc[n,:].isna()].reset_index(drop=True)
			if work_collapsed.size == 0:
				work_collapsed = pd.Series([None])
			unstacked_work_frame[n] = work_collapsed
		unstacked_work = pd.DataFrame.from_dict(unstacked_work_frame, orient='index')
		if quantity == 2:
			out = pd.concat([unstacked_home.iloc[:,0],unstacked_work.iloc[:,0]],axis=1)
		elif quantity > 2:
			for n in range(unstacked.shape[1]):
				unstacked[n][unstacked_home.iloc[:, 0] == unstacked[n]] = None
				unstacked[n][unstacked_work.iloc[:, 0] == unstacked[n]] = None
			others_frame = {}
			for n in unstacked.index:
				others_frame[n] = unstacked.loc[n, :][~unstacked.loc[n, :].isna()].reset_index(drop=True)
			others_frame = pd.DataFrame.from_dict(others_frame,orient='index')
			out = pd.concat([unstacked_home.iloc[:, 0], unstacked_work.iloc[:, 0], others_frame.iloc[:,:quantity-2]], axis=1)
	else:
		out = unstacked.iloc[:,:quantity]
	return out.T.reset_index(drop=True).T


def nighttime_daylight(trajectories_frame,day_start=8,day_end=16,night_start=23,night_end=6):
	"""
	Returns home and work locations based on the most frequently visited places in the night and day.
	:param trajectories_frame: TrajectoriesFrame class object
	:param day_start: the start of the daylight location
	:param day_end: the end of the daylight location
	:param night_start: the start of the nighttime location
	:param night_end: the end of the nighttime location
	:return: Detected locations
	"""
	if not 'labels' in trajectories_frame:
		to_labels(trajectories_frame)
	filtered_trajectories_frame = trajectories_frame[trajectories_frame.geometry.is_valid]
	filtered_trajectories_frame['hod'] = filtered_trajectories_frame.index.get_level_values(1).hour
	daylight_frame = filtered_trajectories_frame[(filtered_trajectories_frame.hod >= day_start) & (filtered_trajectories_frame.hod <= day_end)]
	nighttime_frame = filtered_trajectories_frame[(filtered_trajectories_frame.hod >= night_start) | (filtered_trajectories_frame.hod <= night_end)] #TODO: different time cases
	top_places_day = daylight_frame.groupby([daylight_frame.index.get_level_values(0), 'labels']).count().reset_index()
	top_places_day = pd.DataFrame(top_places_day).groupby('user_id').apply(lambda x: x.sort_values('temp', ascending=False)).reset_index(drop=True)
	top_places_night = nighttime_frame.groupby([nighttime_frame.index.get_level_values(0), 'labels']).count().reset_index()
	top_places_night = pd.DataFrame(top_places_night).groupby('user_id').apply(lambda x: x.sort_values('temp', ascending=False)).reset_index(drop=True)
	merged_day = pd.merge(top_places_day[['user_id', 'labels']], filtered_trajectories_frame.reset_index(), on=['user_id', 'labels'], how='left')[['user_id', 'labels', 'geometry']].drop_duplicates()
	merged_night = pd.merge(top_places_night[['user_id', 'labels']], filtered_trajectories_frame.reset_index(), on=['user_id', 'labels'], how='left')[['user_id', 'labels', 'geometry']].drop_duplicates()
	unstacked_home = merged_day.groupby('user_id')['geometry'].apply(lambda x: x.reset_index(drop=True)).unstack().iloc[:, 0]
	unstacked_night = merged_night.groupby('user_id')['geometry'].apply(lambda x: x.reset_index(drop=True)).unstack().iloc[:, 0]
	return pd.concat([unstacked_home,unstacked_night],axis=1)
	if quantity == 1:
		unstacked = merged.groupby('user_id')['geometry'].apply(lambda x: x.reset_index(drop=True)).unstack().iloc[:,quantity - 1]
	else:
		unstacked = merged.groupby('user_id')['geometry'].apply(lambda x: x.reset_index(drop=True)).unstack().iloc[:,:quantity]
	return unstacked