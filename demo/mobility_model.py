from src.humobi.models.mobility_model.data_sampling import data_sampler
from src.humobi.models.mobility_model.data_generating import data_generator
from src.humobi.structures.trajectory import TrajectoriesFrame
from src.humobi.structures import trajectory as tr
from src.humobi.preprocessing.filters import stop_detection
from src.humobi.tools.user_statistics import *
from src.humobi.tools.processing import start_end
import pandas as pd

from src.humobi.structures import trajectory as tr
from src.humobi.preprocessing.temporal_aggregation import TemporalAggregator
from src.humobi.preprocessing.spatial_aggregation import GridAggregation, ClusteringAggregator, LayerAggregator
from src.humobi.preprocessing.filters import next_location_sequence
from sklearn.cluster import DBSCAN

WEIGHT = False
SIDE = 1000
# path = """Z:\\RJ_DATA\\RIO\\selected_merged\\full_sample_limited_compressed.csv"""
# df_sel = TrajectoriesFrame(path, {
# 	'names': ['id', 'datetime', 'lat', 'lon'], 'crs': "EPSG:4326",
# 	'delimiter': ',', 'skiprows': 1})
# geom_cols = df_sel.geom_cols  # STORE GEOMETRY COLUMN
# crs = df_sel.crs  # STORE CRS
#
# dur = user_trajectories_duration(df_sel, resolution='1D',count_empty=False)
# level1 = set(dur[dur >= 7].index)
# df_sel = df_sel.uloc(level1)
#
# resampled = df_sel.groupby(level=0).resample('1H', level=1)
# resampled_del = resampled.count().iloc[:, 0]
# fracs = resampled_del.groupby(level=0).rolling(24*7).apply(lambda x: (x[x == 0]).count() / (x.count())).reset_index()
# mins = fracs.groupby('user_id').idxmin()
# starts = fracs.groupby('user_id').idxmin()-24*7
# df_selected = pd.concat([resampled.first().iloc[x[0]:y[0]] for x,y in zip(starts.values,mins.values)]).reset_index()
# to_conc = []
# for x in df_selected.groupby('user_id'):
# 	to_conc.append(df_sel.loc[x[0]].loc[x[1].iloc[0].datetime:x[1].iloc[-1].datetime])
# df_sel = pd.concat(to_conc)
# df_sel = TrajectoriesFrame(df_sel.reset_index())
#
#
# frac = fraction_of_empty_records(df_sel, resolution='1H')  # FRACTION OF MISSING RECORDS
# count = count_records(df_sel)  # TOTAL NUMBER OF RECORDS
# count_per_time_frame = count_records_per_time_frame(df_sel, resolution='1D')  # TOTAL NUMBER OF RECORDS PER TIME FRAME
# trajectories_duration = user_trajectories_duration(df_sel, resolution='1H',
#                                                    count_empty=False)  # TOTAL LENGTH OF TRAJECTORIES
# consecutive = consecutive_record(df_sel, resolution='1H')  # THE LONGEST CONSECUTIVE RECORDS IN GIVEN TIME UNIT
#
# SORT BY TIMESTAMP (JUST IN CASE)
# df_sel = df_sel.sort_index(level=[0,1])
#
# print("DATA FILTERED!")
#
# REREAD STRUCTURE
# df_sel = tr.TrajectoriesFrame(df_sel, {'crs': crs, 'geom_cols': geom_cols})

# RUN STOP DETECTION ALGORITHM
# stops = stop_detection(df_sel, distance_condition=1000, time_condition='10min')
# df_sel = stops[stops['is_stop'] == True]

# COMPRESS
# df_sel = start_end(df_sel)
# df_sel = TrajectoriesFrame(df_sel)
# geom_cols = df_sel.geom_cols  # STORE GEOMETRY COLUMNS
# crs = df_sel.crs  # STORE CRS

# df_sel = df_sel.to_crs(dest_crs="EPSG:3857")  # LET'S CONVERT TO METRIC CRS

# DATA AGGREGATION
# NEXT TIME-BIN SEQUENCE TYPE
# DBSCAN-BASED
# df_sel = TrajectoriesFrame("""D:\\dalnloud\\Wyklad1\\full_sample_limited_compressed.csv""",{'names':['user_id','datetime','temp','lat','lon','geometry','is_stop',
#                                                                                     'start','start2','end'],'header':None, 'crs': 4326})
# time_unit = '1H'  # DEFINE TEMPORAL UNIT
# eps = 300  # DEFINE SPATIAL UNIT
# min_pts = 1  # OTHER HYPERPARAMETERS
# clust_agg = LayerAggregator("D:\\Projekty\\4W\\rj_sectors.shp",kwargs={})
# df_sel_lay = clust_agg.aggregate(df_sel)
# clust_agg = ClusteringAggregator(DBSCAN, **{"eps": eps, "min_samples": min_pts})  # DEFINE SPATIAL AGGREGATION ALGORITHM
# df_sel_dbscan = clust_agg.aggregate(df_sel)  # SPATIAL AGGREGATION CALL
# time_agg = TemporalAggregator(time_unit)  # DEFINE TEMPORAL AGGREGATION ALGORITHM
# df_sel_dbscan_time = time_agg.aggregate(df_sel_lay, parallel=True)  # TEMPORAL AGGREGATION CALL
# df_sel_dbscan_time.to_csv("""D:\\Projekty\\4W\\sample_processed.csv""")
df_sel = TrajectoriesFrame("""D:\\Projekty\\4W\\sample_processed.csv""",
                           {'names':['user_id','time','temp','lat','lon','is_stop','start','start2','end','index_right','geometry'],
                            'header':0, 'crs': 4326, 'nrows': 100000, 'usecols': ['user_id','time','temp','lat','lon','geometry']})
dur = user_trajectories_duration(df_sel,'D')[user_trajectories_duration(df_sel,'D') > 7].index
df_sel = df_sel.uloc(dur)
data_sampler(df_sel, "D:\\Projekty\\4W\\census_tracts\\rj_sectors.shp", WEIGHT)
