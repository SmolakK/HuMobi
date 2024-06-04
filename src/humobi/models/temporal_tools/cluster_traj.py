import pandas as pd
import sys, statistics
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
sys.path.append("..")
from src.humobi.models.spatial_tools.misc import rank_freq, normalize_array, nighttime_daylight
from scipy import stats
from collections import Counter
import geopandas as gpd
import os
from itertools import product
import warnings


def extract_unique_combinations(trajectories_frame, aux_cols):
    if aux_cols:
        warnings.warn("Warning: when aux cols are passed, the model is enforced to represent abstract trajectory as a set of probabilities")
        if len(aux_cols) > 1:
            unique_combs = [z for z in product(*[list(pd.unique(trajectories_frame[col].dropna())) for col in aux_cols])]
        else:
            unique_combs = [pd.unique(trajectories_frame[aux_cols[0]].dropna())]
    else:
        unique_combs = []
    return unique_combs


def prepare_abstract_trajectory(grouped, top_places, unique_combs, quantity, length, aux_cols, weights):
    abstract_traj = {}
    for uid, vals in grouped:
        group_cols = ['labels', 'hod'] + (aux_cols if aux_cols else [])
        extract = vals.groupby(group_cols).count().iloc[:, 0].unstack().fillna(0)
        sig_places = []

        for n in range(quantity):
            sig_place = top_places.loc[uid][n]
            if sig_place is not None and sig_place is not np.nan:
                sig_place_label = int(vals[vals['geometry'] == sig_place]['labels'].iloc[0])
                extraction_combs = pd.concat([pd.DataFrame(index=unique_combs), extract.loc[sig_place_label, :]], axis=1).fillna(0).T
                sig_places.append(extraction_combs)
            else:
                sig_places.append(np.zeros((len(unique_combs), length) if aux_cols else (1, length)))

        stacked = np.vstack(sig_places)
        others = extract.sum(0) - stacked.sum(0)
        stacked = np.vstack((stacked, others))
        if weights:
            abstract_traj[uid] = stacked / stacked.sum(axis=0)  # Circadian rhythm
        else:
            abstract_traj[uid] = np.argmax(stacked, axis=0)  # Most commonly visited place at given time
    return abstract_traj

def calculate_distance_matrix(abstract_traj):
    reshaped = np.concatenate([x.reshape(1, -1) for x in abstract_traj.values()], 0)
    cdist = np.zeros((reshaped.shape[0], reshaped.shape[0]))
    for n in range(reshaped.shape[0]):
        for m in range(reshaped.shape[0]):
            cdist[m, n] = stats.wasserstein_distance(reshaped[n, :], reshaped[m, :])
    return cdist

def find_optimal_clustering(cdist):
    tries = {}
    for z in range(1, 5000):  # Find optimal clustering
        eps = z / 1000
        try:
            clust_alg = DBSCAN(metric='precomputed', min_samples=4, eps=eps)
            clust_alg.fit(cdist)
            sh = silhouette_score(cdist, clust_alg.labels_, metric='precomputed')
            tries[eps] = sh
        except:
            pass
    return max(tries, key=tries.get)

def cluster_trajectories(trajectories_frame, top_places = None, length=24, quantity=2, weights=True, clust_alg=DBSCAN(), aux_cols=None):
    """
    Extracts circadian rhythms and clusters users by them.

    :param trajectories_frame: TrajectoriesFrame class object
    :param length: The length of circadian rhythms to extract
    :param quantity: The number of top most important locations to consider
    :param weights: Whether the algorithm should calculate the weight for each of the locations rather than choose the
    most often visited
    :param clust_alg: Clustering algorithm to make clustering
    :param aux_cols: Auxiliary columns in the data with the contextual information
    :return: Clustered circadian rhythms, association of users to clusters, the ratio of users in clusters
    """
    if length <= 24:
        trajectories_frame['hod'] = trajectories_frame.index.get_level_values(1).hour

    unique_combs = extract_unique_combinations(trajectories_frame, aux_cols)
    if top_places is None:
        print("SSSSSSSSSSSSSSSSSS")
        top_places = rank_freq(trajectories_frame, quantity)
    grouped = trajectories_frame.groupby(level=0)

    abstract_traj = prepare_abstract_trajectory(grouped, top_places, unique_combs, quantity, length, aux_cols, weights)
    cdist = calculate_distance_matrix(abstract_traj)
    optimal_eps = find_optimal_clustering(cdist)

    clust_alg = DBSCAN(metric='precomputed', min_samples=4, eps=optimal_eps)
    clust_alg.fit(cdist)
    labels = clust_alg.labels_

    cluster_association = {k: v for k, v in zip(abstract_traj.keys(), labels)}
    abstract_collection = pd.DataFrame(np.concatenate([x.reshape(1, -1) for x in abstract_traj.values()], 0), index=labels)
    cluster_share = Counter([x for x in cluster_association.values() if x != -1])
    cluster_share = {k: v / sum(cluster_share.values()) for k, v in cluster_share.items()}

    return abstract_collection, cluster_association, cluster_share

def circadian_rhythm_extraction(circadian_collection, combs, quantity, length, weighted = False):
	"""
	Calculates average circadian rhythm.
	:param circadian_collection: contains circadian rhythms
	:param combs: all unique aux params combinations
	:param quantity: the number of significant locations
	:param length: the length of circadian rhythm vector
	:return: a dict with clusters as keys and avarage circadian rhythms as values.
	"""
	cluster_indicies = set(circadian_collection.index.difference(set([-1])))
	circadian_rhythm_grouped = {k:v for k,v in zip(cluster_indicies,[circadian_collection.loc[cluster]
																	 for cluster in cluster_indicies])}
	frequency = {}
	for k, v in circadian_rhythm_grouped.items():
		frequency[k] = dict(zip(np.unique(v, return_counts = True)[0],
						 np.unique(v, return_counts = True)[1]))
	if weighted:
		extracted = {k:v.mean() for k,v in circadian_rhythm_grouped.items()}
	elif not weighted:  # TODO: repair
		extracted = {k:v.mode() for k,v in circadian_rhythm_grouped.items()}
		for k,v in extracted.items():
			if len(v) > 1:
				most_freq = max(frequency[k].items(), key = lambda x: x[1])[0]
				v.iloc[0][np.where(np.array(np.isnan(v.iloc[1])) == False)[0][0]] = most_freq
			extracted[k] = v.iloc[:1]
	#Reshaping to aux cols
	new_levels = []
	for x in range(quantity):
		for y in combs:
				new_levels += [(x,y)]
	new_levels += [(-1, 'O')]
	for k in extracted.keys():
		extracted[k] = pd.DataFrame(np.array(extracted[k]).reshape(len(combs)*quantity+1,length))
		extracted[k][['sig','combs']] = pd.DataFrame(new_levels)
		extracted[k] = extracted[k].set_index(['sig','combs'])
	return extracted
