from structures.trajectory import TrajectoriesFrame
from preprocessing.filters import stop_detection
from tools.user_statistics import fraction_of_empty_records, count_records, user_trajectories_duration
from tools.processing import start_end

in_path = """INPUT_PATH"""

df_sel = TrajectoriesFrame(in_path, {'names': ['columns'], 'delimiter': ',', 'skiprows': 1})  # LOAD DATA
geom_cols = df_sel.geom_cols  # ASSIGN GEOMETRY COLUMN
crs = df_sel.geom_cols  # ASSIGN CRS

print("DATA LOADED!")

# FILTRATION
frac = fraction_of_empty_records(df_sel, '1H')
level1 = set(frac[frac < 0.6].index)  # q < 0.6

traj_dur = user_trajectories_duration(df_sel, '1D')
level2 = set(traj_dur[traj_dur > 6].index)  # more than 6 days

counted_records = count_records(df_sel)
level3 = set(counted_records[counted_records >= 100].index)  # at least 100 records

# INDICES SELECTION
selection = level1.intersection(level2)
selection = selection.intersection(level3)
df_sel = df_sel.uloc(list(selection))

# SORT BY TIMESTAMP
df_sel = df_sel.sort_index(level=1)

print("DATA FILTERED!")

# REREAD STRUCTURE
df_sel = TrajectoriesFrame(df_sel, {'crs': crs, 'geom_cols': geom_cols})

# DETECT STOPS
stops = stop_detection(df_sel)
df_sel = stops[stops['is_stop'] == True]
df_sel = df_sel.drop_duplicates()

#COMPRESS
df_sel = start_end(df_sel)

# SAVE DATA
df_sel.to_csv("OUTPUT PATH")
