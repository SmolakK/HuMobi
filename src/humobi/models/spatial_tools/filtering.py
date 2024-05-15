import pandas as pd
import geopandas as gpd


def filter_layer(layer,trajectories):
	"""
	Filters the aggregation layer leaving only cells when an observations are present
	:param layer: Aggregation layer
	:param trajectories: TrajectoriesFrame class object
	:return: A filtered aggregation layer
	"""
	return gpd.sjoin(layer,trajectories,how='inner')[layer.columns].drop_duplicates()