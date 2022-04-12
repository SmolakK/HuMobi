# IMPORTS
import numpy as np
import pandas as pd
from src.humobi.structures.trajectory import TrajectoriesFrame
from tqdm import tqdm
tqdm.pandas()
import concurrent.futures as cf
from itertools import repeat


def fill_gaps(trajectories_frame, fill_method):
	"""
	Filling empty TrajectoriesFrame records

	Args:
		trajectories_frame: TrajectoriesFrame class object
		fill_method: how to fill the data, 'ffill' - forward fill, 'bfill" - backward fill

	Returns:
		Filled TrajectoriesFrame object
	"""
	# TODO: Add more filling routines
	while trajectories_frame['geometry'].isna().any():
		if fill_method == 'ffill':
			trajectories_frame = trajectories_frame.fillna(trajectories_frame.shift())
		if fill_method == 'bfill':
			trajectories_frame = trajectories_frame.fillna(trajectories_frame.shift(-1))
	return trajectories_frame


def next_location_sequence(trajectories_frame):
	"""
	Converts TrajectoriesFrame to the next location sequence

	Args:
		trajectories_frame: TrajectoriesFrame class object

	Returns:
		converted TrajectoriesFrame
	"""
	trajectories_frame = trajectories_frame[~trajectories_frame.isna().iloc[:,0]]
	crs = trajectories_frame.crs
	try:
		geom_cols = trajectories_frame.geom_cols
	except:
		geom_cols = ['lat', 'lon']
	coordinates_frame = trajectories_frame[[geom_cols[0], geom_cols[1]]]
	trajectories_frame = TrajectoriesFrame(
		trajectories_frame[(coordinates_frame != coordinates_frame.shift()).any(axis=1)],
		{'crs': crs, 'geom_cols': geom_cols})
	return trajectories_frame


def _user_stops(indi, single_trajectory, distance_condition, time_condition):
	"""
	Detects stops in a single user's trajectory

	Args:
		indi: user identifier
		single_trajectory: user's trajectory
		distance_condition: distance threshold for stop detection
		time_condition: time threshold for stop detection

	Returns:
		TrajectoriesFrame with records indicated as stops in 'is_stop' column
	"""
	if 'datetime' not in single_trajectory.columns:
		single_trajectory = single_trajectory.reset_index()
	single_trajectory['datetime'] = pd.to_datetime(single_trajectory['datetime'])
	distances = np.array(single_trajectory.to_crs("EPSG:3857"))
	starting_index = 0
	stops = []
	j = 0
	geom_loc = single_trajectory.columns.get_loc('geometry')
	time_loc = single_trajectory.columns.get_loc('datetime')
	while starting_index + j < len(distances) - 1:
		j += 1
		ending_index = starting_index + j
		actual_distance = distances[starting_index][geom_loc].distance(distances[ending_index][geom_loc])
		if actual_distance > distance_condition:
			if ending_index - starting_index > 1:
				if time_condition:
					if distances[ending_index-1][time_loc] - distances[starting_index][time_loc] > pd.Timedelta(time_condition):
						pass
					else:
						starting_index = ending_index
						j = 0
						continue
				stops.append(list(range(starting_index, ending_index)))
			starting_index = ending_index
			j = 0
	single_trajectory['is_stop'] = False
	for stop in stops:
		selected = single_trajectory.iloc[stop]
		mean_lat = selected['lat'].mean()
		mean_lon = selected['lon'].mean()
		single_trajectory.iloc[stop, single_trajectory.columns.get_loc('lat')] = mean_lat
		single_trajectory.iloc[stop, single_trajectory.columns.get_loc('lon')] = mean_lon
		single_trajectory.iloc[stop, single_trajectory.columns.get_loc('is_stop')] = True
	return indi, single_trajectory


def stop_detection(trajectories_frame, distance_condition=300, time_condition='10 min'):
	"""
	Detects all stops in the TrajectoriesFrame. Uses multithreading.

	Args:
		trajectories_frame: TrajectoriesFrame class object
		distance_condition: distance threshold for stop detection
		time_condition: time threshold for stop detection

	Returns:
		TrajectoriesFrame with records indicated as stops in 'is_stop' column
	"""
	result_dic = {}
	with cf.ThreadPoolExecutor() as executor:
		args = [val for indi, val in trajectories_frame.groupby(level=0)]
		ids = [indi for indi, val in trajectories_frame.groupby(level=0)]
		results = list(tqdm(executor.map(_user_stops, ids, args, repeat(distance_condition), repeat(time_condition)),
		                    total=len(ids)))
	for result in results:
		result_dic[result[0]] = result[1]
	detected = pd.concat([x for x in result_dic.values()])
	detected = TrajectoriesFrame(detected)
	return detected
