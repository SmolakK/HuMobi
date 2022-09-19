import pandas as pd
import sys, statistics
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
sys.path.append("..")
from src.humobi.models.spatial_tools.misc import rank_freq, normalize_array, nighttime_daylight
from scipy import stats
from collections import Counter
WEIGHT = False
import geopandas as gpd
import os
from itertools import product
import warnings


def cluster_trajectories(trajectories_frame, length = 24, quantity = 3, weights = True, clust_alg = DBSCAN(),
                         aux_cols = None):
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
	if aux_cols:
		weights = True
		warnings.warn("Warning: when aux cols are passed, the model is enforced to represent abtract trajectory as a set probabilities")
	if len(aux_cols) >1:
		unique_combs = [z for z in product(*[list(pd.unique(trajectories_frame[col].dropna())) for col in aux_cols])]
	else:
		unique_combs = [pd.unique(trajectories_frame[col].dropna()) for col in aux_cols][0]
	top_places = rank_freq(trajectories_frame, quantity = 3)
	abstract_traj = {}
	if length <= 24:
		trajectories_frame['hod'] = trajectories_frame.index.get_level_values(1).hour
	grouped = trajectories_frame.groupby(level=0)
	for uid, vals in grouped:
		extract = vals.groupby(['labels',*aux_cols, 'hod']).count().iloc[:, 0].unstack().fillna(0)
		sig_places = []
		for n in range(quantity):
			sig_place = top_places.loc[uid][n]
			if sig_place is not None:
				sig_place_label = int(vals[vals['geometry'] == sig_place]['labels'].iloc[0])
				extraction_combs = pd.concat([pd.DataFrame(index=unique_combs),extract.loc[sig_place_label,:]],axis=1).fillna(0)
				sig_places.append(extraction_combs)
			else:
				sig_places.append(np.zeros((len(unique_combs),length)))
		stacked = np.vstack(sig_places)
		others = extract.sum(0) - stacked.sum(0)
		stacked = np.vstack((stacked,others))
		if weights:
			abstract_traj[uid] = stacked/stacked.sum(axis=0) #Circadian rhythm
		else:
			abstract_traj[uid] = np.argmax(stacked,axis=0) #most commonly visited place at given time (0-HOME, 1 - WORK, 2 - OTHER for q=2)
	reshaped = np.concatenate([x.reshape(1, -1) for x in abstract_traj.values()], 0) #slices to n hours strips and sets them horizontally in a matrix
	cdist = np.zeros((reshaped.shape[0],reshaped.shape[0]))
	for n in range(reshaped.shape[0]):
		for m in range(reshaped.shape[0]):
			cdist[m,n] = stats.wasserstein_distance(reshaped[n,:],reshaped[m,:])
	tries = {}
	for z in range(1,5000): #find optimal clustering
		eps = z/1000
		try:
			clust_alg = DBSCAN(metric='precomputed',min_samples=4,eps=eps)
			clust_alg.fit(cdist)
			sh = silhouette_score(cdist,clust_alg.labels_,metric='precomputed')
			tries[eps] = sh
		except:
			pass
	eps = max(tries, key=tries.get)
	clust_alg = DBSCAN(metric='precomputed', min_samples=4, eps=eps)
	clust_alg.fit(cdist)
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
