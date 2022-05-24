import pandas as pd
from src.humobi.misc.utils import to_labels


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


def rank_freq(trajectories_frame, quantity=2):
	"""
	Ranks locations visited by each user by the frequency of visits. Used to select the most important location
	:param trajectories_frame: TrajectoriesFrame class object
	:param quantity: The number of location to rank
	:return: Ranked locations
	"""
	if not 'labels' in trajectories_frame:
		to_labels(trajectories_frame)
	filtered_trajectories_frame = trajectories_frame[trajectories_frame.geometry.is_valid]
	top_places = filtered_trajectories_frame.groupby(
		[filtered_trajectories_frame.index.get_level_values(0), 'labels']).count().reset_index()
	top_places = pd.DataFrame(top_places).groupby('user_id').apply(lambda x: x.sort_values('temp', ascending=False)).reset_index(drop=True)
	merged = pd.merge(top_places[['user_id', 'labels']], filtered_trajectories_frame.reset_index(), on=['user_id', 'labels'], how='left')[['user_id', 'labels', 'geometry']].drop_duplicates()
	if quantity == 1:
		unstacked = merged.groupby('user_id')['geometry'].apply(lambda x: x.reset_index(drop=True)).unstack().iloc[:,quantity - 1]
	else:
		unstacked = merged.groupby('user_id')['geometry'].apply(lambda x: x.reset_index(drop=True)).unstack().iloc[:,:quantity]
	return unstacked
