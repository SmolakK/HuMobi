from humobi.structures import trajectory as tr
from humobi.measures.individual import *
from humobi.measures.collective import *

# LOAD DATA
in_path = "aggregated_sample.csv"
df_sel = tr.TrajectoriesFrame(in_path, {'delimiter': ',', 'crs': 3857})  # LOAD DATA
geom_cols = df_sel.geom_cols  # STORE GEOMETRY COLUMNS
crs = df_sel.crs  # STORE CRS

# INDIVIDUAL METRICS
distinct_total = num_of_distinct_locations(df_sel)
vfreq = visitation_frequency(df_sel)
distinct_over_time = distinct_locations_over_time(df_sel, resolution='1H', reaggregate=False)
jump = jump_lengths(df_sel)
trips = nonzero_trips(df_sel)
st = self_transitions(df_sel)
wt = waiting_times(df_sel)
mc = center_of_mass(df_sel)
rog = radius_of_gyration(df_sel, time_evolution=False)
rog_time = radius_of_gyration(df_sel, time_evolution=True)
msd = mean_square_displacement(df_sel, time_evolution=False)
msd_time = mean_square_displacement(df_sel, time_evolution=True)
rt = return_time(df_sel)
rt_place = return_time(df_sel, by_place=True)
ran_ent = random_entropy(df_sel)
unc_ent = unc_entropy(df_sel)
# real_ent = real_entropy(df_sel)
random_pred = random_predictability(df_sel)
unc_pred = unc_predictability(df_sel)
# real_pred = real_predictability(df_sel)
stat = stationarity(df_sel)
regul = regularity(df_sel)

# COLLECTIVE METRICS
dist = dist_travelling_distance(df_sel)
pairwise_flows = flows(df_sel, flows_type='all')

