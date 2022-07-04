import pandas as pd
import sys, statistics
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
sys.path.append("..")
from humobi.models.spatial_tools.misc import rank_freq, normalize_array
from scipy import stats
from collections import Counter
WEIGHT = False
import geopandas as gpd
import os

def cluster_trajectories(trajectories_frame, length = 24, quantity = 2, weights = True, clust_alg = KMeans(2),
                         aux_data = None, aux_col = None, aux_folder = None):
	"""
	Extracts circadian rhythms and clusters users by them.
	:param trajectories_frame: TrajectoriesFrame class object
	:param length: The lenght of circadian rhythms to extract
	:param quantity: The number of top most important locations to consider
	:param weights: Whether the algorithm should calculate the weight for each of the locations rather than choose the
	most often visited
	:param clust_alg: Clustering algorithm to make clusterization
	:return: Clustered circadian rhythms, association of users to clusters, the ratio of users in clusters
	"""
	top_places = rank_freq(trajectories_frame, quantity)
	abstract_traj = {}
	if length <= 24:
		trajectories_frame['hod'] = trajectories_frame.index.get_level_values(1).hour
	grouped = trajectories_frame.groupby(level=0)
	for uid, vals in grouped:
		extract = vals.groupby(['labels', 'hod']).count().iloc[:, 0].unstack().fillna(0)
		sig_places = []
		for n in range(quantity):
			sig_place = top_places.loc[uid][n]
			if sig_place is not None:
				sig_place_label = int(vals[vals['geometry'] == sig_place]['labels'].iloc[0])
				sig_places.append(extract.loc[sig_place_label,:].values)
			else:
				sig_places.append(np.zeros((1,length)))
		stacked = np.vstack(sig_places)
		others = extract.sum(0) - stacked.sum(0)
		stacked = np.vstack((stacked,others))
		if weights:
			abstract_traj[uid] = stacked/stacked.sum(axis=0) #Circadian rhythm
		else:
			abstract_traj[uid] = np.argmax(stacked,axis=0) #most commonly visited place at given time (0-HOME, 1 - WORK, 2 - OTHER for q=2)
	reshaped = np.concatenate([x.reshape(1, -1) for x in abstract_traj.values()], 0)
	clust_alg.fit(reshaped)
	labels = clust_alg.labels_
	cluster_association = {k: v for k, v in zip(abstract_traj.keys(), clust_alg.labels_)}
	abstract_collection = pd.DataFrame(reshaped, index=labels)
	cluster_share = Counter([x for x in cluster_association.values() if x != -1])
	cluster_share = {k: v / sum(cluster_share.values()) for k, v in cluster_share.items()}
	return abstract_collection, cluster_association, cluster_share


def circadian_rhythm_extraction(circadian_collection):
	"""
	Calculates average circadian rhythm.
	:param circadian_collection: contains circadian rhythms
	:return: a dict with clusters as keys and avarage circadian rhythms as values.
	"""
	cluster_indicies = set(circadian_collection.index.difference(set([-1])))
	circadian_rhythm_grouped = {k:v for k,v in zip(cluster_indicies,[circadian_collection.loc[cluster]
																	 for cluster in cluster_indicies])}
	frequency = {}
	for k, v in circadian_rhythm_grouped.items():
		frequency[k] = dict(zip(np.unique(v, return_counts = True)[0],
						 np.unique(v, return_counts = True)[1]))
	if WEIGHT:
		extracted = {k:v.mean() for k,v in circadian_rhythm_grouped.items()}
	elif not WEIGHT:
		extracted = {k:v.mode() for k,v in circadian_rhythm_grouped.items()}
		for k,v in extracted.items():
			if len(v) > 1:
				most_freq = max(frequency[k].items(), key = lambda x: x[1])[0]
				v.iloc[0][np.where(np.array(np.isnan(v.iloc[1])) == False)[0][0]] = most_freq
			extracted[k] = v.iloc[:1]
	return extracted
