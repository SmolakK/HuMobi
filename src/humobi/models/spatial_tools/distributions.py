import pandas as pd
import geopandas as gpd
import sys

sys.path.append("..")
from misc.utils import normalize
from models.spatial_tools.misc import rank_freq


def calculate_distances(gs1, gs2):
	"""
	Calculates the distance (ellipsoidal) between to GeoSeries
	:param gs1: GeoSeries1
	:param gs2: GeoSeries2
	:return: The GeoSeries of distances
	"""
	return gs1.distance(gs2)


def commute_distances(trajectories_frame, quantity=2):
	"""
	Calculates the commuting distances for each user. Quantity regulates the number of locations to which the distance
	is calculated.
	:param trajectories_frame: TrajectoriesFrame class object
	:param quantity: The number of locations to which the distance will be calculated
	:return: The DataFrame of distances to the locations
	"""
	sig_frame = rank_freq(trajectories_frame, quantity=quantity)
	indices = []
	distances = {}
	for n in range(quantity - 1):
		for k in range(n + 1, quantity):
			indices.append((n, k))
	indices = sorted(indices, key=lambda x: x[1])
	prev_k = 1
	df_list = []
	for n, k in indices:
		if k != prev_k:
			distances[prev_k] = pd.concat(df_list, axis=1)
			df_list = []
		prev_k = k
		gs1 = gpd.GeoDataFrame(sig_frame.iloc[:, n], geometry=n)
		gs2 = gpd.GeoDataFrame(sig_frame.iloc[:, k], geometry=k)
		distance = calculate_distances(gs1, gs2)
		df_list.append(pd.DataFrame([distance, sig_frame.iloc[:, n]]).T)  # distance and starting point geometry
	distances[k] = pd.concat(df_list, axis=1)
	return distances


def commute_distances_to_2d_distribution(commute_distances_frame, layer, crs="epsg:3857", return_centroids=False):
	"""
	Converts commuting distances into 2D distribution of median commuting distances.
	:param commute_distances_frame: DataFrame with commuting distances to important locations
	:param layer: Aggregation layer
	:param crs: CRS of output data
	:param return_centroids: Whether full layer or centroids should be returned
	:return: 2D distribution of median commuting distances
	"""
	commute_distributions = {}
	for k,v in commute_distances_frame.items():
		commute_distances_frame = gpd.GeoDataFrame(v.reset_index(drop=True), geometry=0)
		if crs:
			pass
		else:
			crs = layer.crs
		grouped = gpd.tools.sjoin(commute_distances_frame, layer, how='right').groupby(level=0)
		to_concat = []
		for uid, vals in grouped:
			if vals.iloc[:, 1:k+1].isna().all()[0]:
				pass
			else:
				vals.iloc[0, 1:k+1] = vals.iloc[:, 1:k+1].median()
			to_concat.append(vals.iloc[0, :])
		merged = pd.concat(to_concat, axis=1).T.iloc[:, 1:3+k]
		merged = merged.rename(columns={"Unnamed 0": 'distance'})
		merged = gpd.GeoDataFrame(merged, geometry='geometry').fillna(0)
		if return_centroids:
			merged['geometry'] = merged['geometry'].centroid
		commute_distributions[k] = merged
	return commute_distributions


def convert_to_2d_distribution(significant_places_frame, layer, quantity=1, crs="epsg:3857", return_centroids=False):
	"""
	A general function for converting points into 2D distribution
	:param significant_places_frame: Points layer - usually significant places
	:param layer: Aggregation layer
	:param quantity: The number of layers to return - if there is than one point per index, more layers can be returned
	:param crs: CRS of output data
	:param return_centroids: Whether full layer or centroids should be returned
	:return: List of DataFrames with 2D distributions
	"""
	distributions = []
	significant_places_frame = significant_places_frame.reset_index()
	if crs:
		pass
	else:
		crs = layer.crs
	if isinstance(significant_places_frame, pd.DataFrame):
		for to_convert in range(quantity):
			col_name = significant_places_frame.columns[to_convert + 1]
			sig_locs = significant_places_frame.loc[:, col_name]
			sig_locs = sig_locs.dropna()
			if sig_locs.dtype == 'geometry':
				sig_locs_data = gpd.GeoDataFrame(sig_locs, crs=crs, geometry=col_name)
			elif sig_locs.dtype == 'str':
				sig_locs = sig_locs.str.split(";", expand=True, ).astype(float)
				sig_locs_data = gpd.GeoDataFrame(significant_places_frame,
				                                 geometry=gpd.points_from_xy(*[sig_locs[x] for x in sig_locs]), crs=crs)
			account = gpd.tools.sjoin(sig_locs_data, layer).groupby('index_right').count().iloc[:, 0]
			layer_counted = pd.merge(layer, account, left_index=True, right_index=True, how='outer').fillna(0)
			layer_counted = layer_counted.rename(columns={to_convert: 'counted'})
			layer_counted = normalize(layer_counted, 'counted')
			if return_centroids:
				layer_counted['geometry'] = layer_counted['geometry'].centroid
			distributions.append(layer_counted)
	return distributions