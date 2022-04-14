from humobi.structures import trajectory as tr
from humobi.preprocessing.filters import stop_detection
from humobi.tools.user_statistics import *
from humobi.tools.processing import start_end

in_path = """converted_sample.csv"""
df_sel = tr.TrajectoriesFrame(in_path, {'crs': 4326})  # ALREADY CONVERTED (SEE data_reading.py demo)
geom_cols = df_sel.geom_cols  # STORE GEOMETRY COLUMN
crs = df_sel.crs  # STORE CRS

print("DATA LOADED!")

# DATA SELECTION METHODS (YOU CAN USE STANDARD PANDAS METHODS TO SELECT DATA TOO)
# USER ID SELECTION
# GET ONE USER BY ID
one_user = df_sel.uloc(0)

users_list = df_sel.get_users()  # LIST OF ALL IDS

# GET MULTIPLE USERS BY ID
many_users = df_sel.uloc(users_list)

# USER STATISTICS

frac = fraction_of_empty_records(df_sel, resolution='1H')  # FRACTION OF MISSING RECORDS
count = count_records(df_sel)  # TOTAL NUMBER OF RECORDS
count_per_time_frame = count_records_per_time_frame(df_sel, resolution='1D')  # TOTAL NUMBER OF RECORDS PER TIME FRAME
trajectories_duration = user_trajectories_duration(df_sel, resolution='1H',
                                                   count_empty=False)  # TOTAL LENGTH OF TRAJECTORIES
consecutive = consecutive_record(df_sel, resolution='1H')  # THE LONGEST CONSECUTIVE RECORDS IN GIVEN TIME UNIT

# FILTRATION WITH USER STATISTICS
frac = fraction_of_empty_records(df_sel, '1H')
level1 = set(frac[frac < 0.9].index)  # FRACTION OF MISSING RECORDS < 0.6

traj_dur = user_trajectories_duration(df_sel, '1D')
level2 = set(traj_dur[traj_dur > 6].index)  # MORE THAN 6 DAYS OF DATA

counted_records = count_records(df_sel)
level3 = set(counted_records[counted_records >= 100].index)  # AT LEAST 100 RECORDS IN TOTAL

# INDICES SELECTION
selection = level1.intersection(level2)
selection = selection.intersection(level3)
df_sel = df_sel.uloc(list(selection))  # USER FILTRATION WITH ULOC METHOD

# SORT BY TIMESTAMP (JUST IN CASE)
df_sel = df_sel.sort_index(level=1)

print("DATA FILTERED!")

# REREAD STRUCTURE
df_sel = tr.TrajectoriesFrame(df_sel, {'crs': crs, 'geom_cols': geom_cols})

# RUN STOP DETECTION ALGORITHM
stops = stop_detection(df_sel, distance_condition=300, time_condition='10min')
df_sel = stops[stops['is_stop'] == True]
df_sel = df_sel.drop_duplicates()

# COMPRESS
df_sel = start_end(df_sel)

# SAVE DATA
df_sel.to_csv("preprocessed_stops_sample.csv")
