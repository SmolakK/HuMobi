import sys

sys.path.append("..")
from src.humobi.structures.trajectory import TrajectoriesFrame
from src.humobi.models.temporal_tools import cluster_traj
from src.humobi.models.spatial_tools import misc, distributions, generating, filtering
from src.humobi.misc import create_grid
from src.humobi.models.spatial_tools.misc import rank_freq
from math import ceil
from src.humobi.models.agent_module.generate_agents import generate_agents
from src.humobi.models.spatial_modules.where import where
from src.humobi.models.temporal_tools.when import when
from datetime import timedelta as td, datetime
import pandas as pd

WEIGHT = False
SIDE = 1000

path = r"D:\Projekty\bias\london\london_1H_61.12331451408496_1.csv"
trajectories_frame = TrajectoriesFrame(path, {
    'names': ['user_id', 'datetime', 'temp', 'lat', 'lon', 'labels', 'start', 'end', 'geometry'], 'crs': "EPSG:3857",
    'delimiter': ',', 'skiprows': 1, 'nrows': 10_000})

#SAMPLING DATA FROM USERS
circadian_collection, cluster_association, cluster_share, unique_combs = cluster_traj.cluster_trajectories(
    trajectories_frame, weights=WEIGHT)
commute_dist = distributions.commute_distances(trajectories_frame, quantity=2)
layer = create_grid.create_grid(trajectories_frame, resolution=SIDE)
layer = filtering.filter_layer(layer, trajectories_frame)
unique_labels = set(cluster_association.values()).difference(set([-1]))
sig_frame = rank_freq(trajectories_frame, quantity=2)
cluster_spatial_distributions = {}
cluster_commute_distributions = {}
for n in [unique_labels][0]:
    group_indicies = [k for k, v in cluster_association.items() if v == n]
    group_sig_frame = sig_frame.loc[group_indicies]
    group_commute_dist = {k: v.loc[group_indicies] for k, v in commute_dist.items()}
    dist_list = distributions.convert_to_2d_distribution(sig_frame, layer, crs="EPSG:3857", return_centroids=True,
                                                         quantity=2)
    commute_distributions = distributions.commute_distances_to_2d_distribution(group_commute_dist, layer,
                                                                               crs="EPSG:3857", return_centroids=True)
    cluster_spatial_distributions[n] = dist_list
    cluster_commute_distributions[n] = commute_distributions

# GENERATING DATA FOR AGENTS
to_generate = 29
for label, share in cluster_share.items():
    amount = ceil(share * to_generate)
    current_spatial_distributions = cluster_spatial_distributions[label]
    current_commute_distributions = cluster_commute_distributions[label]
    home_positions = generating.generate_points_from_distribution(current_spatial_distributions[0], amount)
    work_positions = generating.select_points_with_commuting(home_positions, current_spatial_distributions,
                                                             current_commute_distributions)
    activity_areas = generating.generate_activity_areas('ellipse', home_positions, work_positions, layer, 1.0)
    circadian_rhythm = cluster_traj.circadian_rhythm_extraction(circadian_collection,[],2,24)
    circadian_rhythm = circadian_rhythm[0]  # This is temporary
    agents = generate_agents(amount, label, home_positions, work_positions, activity_areas, circadian_rhythm)

# MOVING AGENTS IN THE SIMULATION
sim_start = '01-01-2020'
end_sim = '10-01-2020'
sim_start = datetime.strptime(sim_start, '%d-%m-%Y')
end_sim = datetime.strptime(end_sim, '%d-%m-%Y')
timeslots = []
for i in range((end_sim - sim_start).days * 24 + 1):
    timeslots.append(sim_start + td(hours=i))

for timeslot in timeslots:
    for agent in agents:
        when(agent, timeslot)
        where(agent)

zipped_history = [pd.DataFrame(zip(agent.history,timeslots),columns=['geometry','datetime']) for agent in agents]
output = pd.concat([y.assign(user_id=x) for x,y in enumerate(zipped_history)])
