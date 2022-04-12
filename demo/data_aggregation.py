from humobi.structures import trajectory as tr
from humobi.preprocessing.temporal_aggregation import TemporalAggregator
from humobi.preprocessing.spatial_aggregation import GridAggregation, ClusteringAggregator
from humobi.preprocessing.filters import next_location_sequence
from sklearn.cluster import DBSCAN

in_path = "preprocessed_stops_sample.csv"  # STOPS DETECTED
df_sel = tr.TrajectoriesFrame(in_path, {'delimiter': ',', 'crs': 4326})  # LOAD DATA
geom_cols = df_sel.geom_cols  # STORE GEOMETRY COLUMNS
crs = df_sel.crs  # STORE CRS

df_sel = df_sel.to_crs(dest_crs="EPSG:3857")  # LET'S CONVERT TO METRIC CRS

# DATA AGGREGATION
# NEXT TIME-BIN SEQUENCE TYPE
# DBSCAN-BASED
time_unit = '1H'  # DEFINE TEMPORAL UNIT
eps = 300  # DEFINE SPATIAL UNIT
min_pts = 2  # OTHER HYPERPARAMETERS

clust_agg = ClusteringAggregator(DBSCAN, **{"eps": eps, "min_samples": min_pts})  # DEFINE SPATIAL AGGREGATION ALGORITHM
df_sel_dbscan = clust_agg.aggregate(df_sel)  # SPATIAL AGGREGATION CALL
time_agg = TemporalAggregator(time_unit)  # DEFINE TEMPORAL AGGREGATION ALGORITHM
df_sel_dbscan_time = time_agg.aggregate(df_sel_dbscan, parallel=True)  # TEMPORAL AGGREGATION CALL

# GRID-BASED
gird_resolution = 1000  # DEFINE SPATIAL UNIT (GRID)
time_unit = '1H'  # DEFINE TEMPORAL UNIT
grid_agg = GridAggregation(gird_resolution)  # DEFINE GRID AGGREGATION ALGORITHM
df_sel_grid = grid_agg.aggregate(df_sel, parralel=False)  # CALL AGGREGATION
time_agg = TemporalAggregator(time_unit)  # DEFINE TEMPORAL AGGREGATION ALGORITHM
df_sel_grid_time = time_agg.aggregate(df_sel_grid)  # AGGREGATION CALL

# NEXT LOCATION SEQUENCES TYPE
df_sel_copy = tr.TrajectoriesFrame(df_sel, {'crs': 3857})  # MAKE TR COPY

# DBSCAN BASED
time_unit = '1H'  # DEFINE TEMPORAL UNIT
eps = 300  # DEFINE SPATIAL UNIT
min_pts = 2
clust_agg = ClusteringAggregator(DBSCAN, **{"eps": eps, "min_samples": min_pts})
df_sel_dbscan = clust_agg.aggregate(df_sel_copy)
df_sel_time_seq = next_location_sequence(df_sel_dbscan_time)  # CREATE NEXT LOCATION SEQUENCES

# GRID BASED
gird_resolution = 1000
grid_agg = GridAggregation(gird_resolution)
df_sel_grid_time = grid_agg.aggregate(df_sel_copy)
df_sel_time_seq = next_location_sequence(df_sel_grid_time)  # CREATE NEXT LOCATION SEQUENCES
