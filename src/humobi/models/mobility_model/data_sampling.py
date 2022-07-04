import sys
sys.path.append("..")
from humobi.misc import create_grid
from humobi.structures.trajectory import TrajectoriesFrame
from humobi.models.temporal_tools import cluster_traj
from humobi.models.spatial_tools import misc, distributions, generating, filtering
from humobi.models.spatial_tools.misc import rank_freq
from math import ceil
from humobi.models.agent_module.generate_agents import generate_agents
from humobi.preprocessing.spatial_aggregation import LayerAggregator


def data_sampler(input_data, aggregation, weigthed, aux_data = None, aux_folder = None):
	"""
	:param input_data: 
	:return: 
	"""
	if input_data.crs.coordinate_system.name == 'ellipsoidal':
		input_data = input_data.to_crs("epsg:3857")
	if isinstance(aggregation,int):
		layer = create_grid.create_grid(input_data, resolution = aggregation)
	elif isinstance(aggregation,str):
		layer = LayerAggregator(aggregation)
		if not layer.layer.crs == input_data.crs:
			layer.layer = layer.layer.to_crs(input_data.crs)
	layer = filtering.filter_layer(layer,input_data)
	circadian_collection, cluster_association, cluster_share = cluster_traj.cluster_trajectories(input_data, weights=weigthed, quantity=2)
	commute_dist = distributions.commute_distances(input_data, quantity = 2)
	unique_labels = set(cluster_association.values()).difference(set([-1]))
	sig_frame = rank_freq(input_data, quantity = 2)
	cluster_spatial_distributions = {}
	cluster_commute_distributions = {}
	for n in [unique_labels][0]:
		group_indicies = [k for k,v in cluster_association.items() if v == n]
		group_sig_frame = sig_frame.loc[group_indicies]
		group_commute_dist = {k:v.loc[group_indicies] for k,v in commute_dist.items()}
		dist_list = distributions.convert_to_2d_distribution(group_sig_frame, layer, return_centroids=True, quantity = 2)
		commute_distributions = distributions.commute_distances_to_2d_distribution(group_commute_dist, layer, return_centroids=True)
		cluster_spatial_distributions[n] = dist_list
		cluster_commute_distributions[n] = commute_distributions
	to_generate = 217
	generated_agents = []
	for label, share in cluster_share.items():
		amount = ceil(share*to_generate)
		current_spatial_distributions = cluster_spatial_distributions[label]
		current_commute_distributions = cluster_commute_distributions[label]
		home_positions = generating.generate_points_from_distribution(current_spatial_distributions[0], amount)
		work_positions = generating.select_points_with_commuting(home_positions,current_spatial_distributions[1],current_commute_distributions, spread=.05)
		activity_areas = generating.generate_activity_areas('ellipse', home_positions, work_positions, layer, 1.0)
		agents = generate_agents(amount, label, home_positions, work_positions, activity_areas)
		generated_agents += agents
	circadian_rhythm = cluster_traj.circadian_rhythm_extraction(circadian_collection)