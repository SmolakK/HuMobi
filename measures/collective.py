import sys
sys.path.append("..")
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from measures.individual import jump_lengths
from preprocessing.filters import next_location_sequence


def dist_travelling_distance(trajectories_frame):
	"""
	Calculates the distribution of travelling distances for each user
	:param trajectories_frame: TrajectoriesFrame class object
	:return: Traveling distances for each user
	"""
	jumps = jump_lengths(trajectories_frame).dropna().droplevel([1,2])
	return jumps[jumps != 0]


def flows(trajectories_frame, flows_type='all'):
	"""
	Calculates the number of flows for each aggregation cell. All flows, only incoming or only outgoing flows can be
	counted.
	:param trajectories_frame: TrajectoriesFrame class object
	:param flows_type: Type of flows to be counted (possible values: 'all', 'incoming', 'outgoing') (default = 'all')
	:return: a DataFrame with flows grouped by cells of aggregation grid
	"""
	next_locations = next_location_sequence(trajectories_frame)
	next_locations['geometry'] = next_locations['geometry'].astype(str)
	double_flows = next_locations.groupby(level=0).progress_apply(lambda x: x[1:-1].groupby('geometry').count()).\
		droplevel(0)
	if flows_type == 'all':
		double_flows = double_flows*2
	if flows_type == 'incoming':
		trajectories_edges = [-1]
	elif flows_type == 'outgoing':
		trajectories_edges = [0]
	else:
		trajectories_edges = [0, -1]
	single_flows = next_locations.groupby(level=0).progress_apply(lambda x: x.iloc[trajectories_edges].groupby('geometry').count()).\
		droplevel(0)
	total_flows = pd.concat([double_flows,single_flows])
	total_flows = total_flows[total_flows.columns[0]]
	return total_flows.groupby('geometry').sum().sort_values(ascending=False)
