from structures.trajectory import TrajectoriesFrame
from preprocessing.temporal_aggregation import TemporalAggregator
from preprocessing.spatial_aggregation import GridAggregation, ClusteringAggregator
from preprocessing.filters import next_location_sequence
import numpy as np
from sklearn.cluster import DBSCAN
import os
import pandas as pd

save_dir = "OUTPUT_DIR"
in_path = "INPUT_PATH"  # STOPS DETECTED
df_sel = TrajectoriesFrame(in_path, {'names': ['columns'], 'delimiter': ',', 'skiprows': 1})  # LOAD DATA
geom_cols = df_sel.geom_cols
crs = df_sel.geom_cols
df_sel = df_sel.drop_duplicates()

# DATA AGGREGATION
# DBSCAN-BASED
time_units = ['5min', '10min', '15min', '30min', '45min', '1H', '6H', '12H', '1D', '2D', '3D', '6D',
              '12D']  # DEFINE TEMPORAL UNITS
eps_space = np.logspace(1, 4.8, 30, endpoint=True)  # DEFINE SPATIAL UNITS (HYPERPARAMETERS)
pts_space = [1, 2, 4]  # OTHER HYPERPARAMETERS

# NEXT TIME-BIN SEQUENCES CREATION

# CALCULATE DIFFERENT VARIANTS OF AGGREGATION
for eps_step in eps_space:
	for samples_step in pts_space:
		if not df_sel.groupby(level=0).apply(lambda x: len(x) > samples_step).all():  # SKIP IF NOT ENOUGH DATA
			continue
		clust_agg = ClusteringAggregator(DBSCAN, **{"eps": eps_step,
		                                            "min_samples": samples_step})  # DEFINE SPATIAL AGGREGATION ALGORITHM
		df_sel_dbscan = clust_agg.aggregate(df_sel)  # AGGREGATION CALL
		for time_unit in time_units:
			time_agg = TemporalAggregator(time_unit)  # DEFINE TEMPORAL AGGREGATION AGLORITHM
			df_sel_dbscan_time = time_agg.aggregate(df_sel_dbscan, parralel=True)  # AGGREGATION CALL
			path_write = os.path.join(save_dir, "PREFIX_" + time_unit + "_" + str(eps_step) + "_" + str(
				samples_step) + ".csv")  # OUTPUT PATH
			df_sel_dbscan_time.to_csv(path_write)  # WRITE DOWN

gird_resolutions = np.logspace(1, 4.8, 30, endpoint=True)  # DEFINE SPATIAL UNITS (GRID)
time_units = ['5min', '10min', '15min', '30min', '45min', '1H', '6H', '12H', '1D', '2D', '3D', '6D',
              '12D']  # DEFINE TEMPORAL UNITS
for grid_unit in gird_resolutions:
	grid_agg = GridAggregation(grid_unit)  # DEFINE GRID AGGREGATION ALGORITHM
	df_sel_grid = grid_agg.aggregate(df_sel, parralel=False)  # CALL AGGREGATION
	for time_unit in time_units:
		time_agg = TemporalAggregator(time_unit)  # DEFINE TEMPORAL AGGREGATION ALGORITHM
		df_sel_grid_time = time_agg.aggregate(df_sel_grid)  # AGGREGATION CALL
		path_write = os.path.join(save_dir,
		                          "PREFIX_" + time_unit + "_" + str(grid_unit) + "_" + "GRID" + ".csv")  # OUTPUT PATH
		df_sel_dbscan_time.to_csv(path_write)  # WRITE DOWN

# NEXT LOCATION SEQUENCES CREATION - COMMENTS THE SAME AS ABOVE
df_sel_time = TrajectoriesFrame(df_sel, {'geom_cols': geom_cols})

eps_space = np.logspace(1, 4.8, 30, endpoint=True)  # DEFINE SPATIAL UNITS (HYPERPARAMETERS)
pts_space = [1, 2, 4]  # OTHER HYPERPARAMETERS
for eps_step in eps_space:
	for samples_step in pts_space:
		if not df_sel_time.groupby(level=0).apply(lambda x: len(x) > samples_step).all():
			continue
		clust_agg = ClusteringAggregator(DBSCAN, **{"eps": eps_step, "min_samples": samples_step})
		df_sel_dbscan_time = clust_agg.aggregate(df_sel_time)
		df_sel_time_seq = next_location_sequence(df_sel_dbscan_time)  # CREATE NEXT LOCATION SEQUENCES
		path_write = os.path.join(save_dir, "PREFIX_" + "seq" + "_" + str(eps_step) + "_" + str(
			samples_step) + ".csv")
		df_sel_dbscan_time.to_csv(path_write)

gird_resolutions = np.logspace(1, 4.8, 30, endpoint=True)
for grid_unit in gird_resolutions:
	grid_agg = GridAggregation(grid_unit)
	df_sel_grid_time = grid_agg.aggregate(df_sel_time)
	df_sel_time_seq = next_location_sequence(df_sel_grid_time)  # CREATE NEXT LOCATION SEQUENCES
	path_write = os.path.join(save_dir, "PREFIX_" + "seq" + "_" + str(grid_unit) + "_" + "GRID" + ".csv")
	df_sel_time_seq.to_csv(path_write)
