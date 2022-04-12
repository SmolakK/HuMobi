from src.humobi.structures.trajectory import TrajectoriesFrame
import sys
sys.path.append("..")
from src.humobi.models.temporal_tools import cluster_traj
from src.humobi.models.spatial_tools import misc, distributions, generating, filtering
from src.humobi.models.mobility_model import data_generating
from src.humobi.misc import create_layer
from src.humobi.models.spatial_tools.misc import rank_freq
from math import ceil
from src.humobi.models.agent_module.generate_agents import generate_agents


WEIGHT = False
SIDE = 1000


path = "C:\\Users\\ppp\\Desktop\\staz\\repo-HuMobi2\\HuMobi2\\oryginal.csv"
trajectories_frame = TrajectoriesFrame(path, {
	'names': ['id', 'datetime', 'temp', 'lat', 'lon', 'labels', 'start', 'end', 'geometry'], 'crs': "EPSG:27700",
	'delimiter': ',', 'skiprows': 1, 'nrows': 9386})

circadian_collection, cluster_association, cluster_share = cluster_traj.cluster_trajectories(trajectories_frame, weights=WEIGHT)
commute_dist = distributions.commute_distances(trajectories_frame, quantity = 2)
layer = create_layer.create_grid(trajectories_frame, resolution = SIDE)
layer = filtering.filter_layer(layer,trajectories_frame)
unique_labels = set(cluster_association.values()).difference(set([-1]))
sig_frame = rank_freq(trajectories_frame, quantity = 2)
cluster_spatial_distributions = {}
cluster_commute_distributions = {}
for n in [unique_labels][0]:
	group_indicies = [k for k,v in cluster_association.items() if v == n]
	group_sig_frame = sig_frame.loc[group_indicies]
	group_commute_dist = {k:v.loc[group_indicies] for k,v in commute_dist.items()}
	dist_list = distributions.convert_to_2d_distribution(sig_frame, layer, crs="epsg:27700", return_centroids=True, quantity = 2)
	commute_distributions = distributions.commute_distances_to_2d_distribution(group_commute_dist, layer, crs="epsg:27700", return_centroids=True)
	cluster_spatial_distributions[n] = dist_list
	cluster_commute_distributions[n] = commute_distributions
to_generate = 29
generated_agents = []
for label, share in cluster_share.items():
	amount = ceil(share*to_generate)
	current_spatial_distributions = cluster_spatial_distributions[label]
	current_commute_distributions = cluster_commute_distributions[label]
	home_positions = generating.generate_points_from_distribution(current_spatial_distributions[0], amount)
	work_positions = generating.select_points_with_commuting(home_positions,current_spatial_distributions[1],current_commute_distributions)
	activity_areas = generating.generate_activity_areas('ellipse', home_positions, work_positions, layer, 1.0)
	agents = generate_agents(amount, label, home_positions, work_positions, activity_areas)
	generated_agents += agents
circadian_rhythm = cluster_traj.circadian_rhythm_extraction(circadian_collection)