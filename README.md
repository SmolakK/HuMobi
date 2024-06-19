# HuMobi

<b> HuMobi is going now through a major update! If you face any troubles, be sure we are working on removing these bugs right now (dev team) </b>

You can find an interactive version of this README on Google Colab here: https://colab.research.google.com/drive/1nmYJDBYr1nN89LLx2NKP6Q5bqQ2mugbO?usp=sharing
 
 ## Table of contents
* [General info](#General-info)
* [Installing HuMobi](#Installing-HuMobi)
* [Data reading](#Data-reading)
* [Data preprocessing](#Data-preprocessing)
* [Metrics](#Metrics)
* [Data generation routines](#Data-generation-routines)
* [Next location predictions](#Next-location-predictions)
* [Paper: Explaining human mobility predictions through pattern matching algorithm](#Paper-Explaining-human-mobility-predictions-through-pattern-matching-algorithm)
* [Known issues](#Known-issues)
 
## General Info
This is the HuMobi library. It is a dedicated Python library for human mobility data processing, which mostly extends Pandas
DataFrames and Geopandas GeoDataFrames and facilitates operating on a very specific data structure of individual mobility trajectories. Below you will find info on how to install HuMobi library on your computer and a set of demos covering most of the library functionalities.

This library is mainly devoted to processing individual mobility trajectories and focuses on human mobility prediction and modelling. Initially, it was implemented during PhD studies of Kamil Smolak, a PhD candidate at the Wrocław University of Environmental and Life Sciences.

If you use this library please cite the below work:
```
Smolak, K., Siła-Nowicka, K., Delvenne, J. C., Wierzbiński, M., & Rohm, W. (2021). The impact of human mobility data scales and processing on movement predictability. Scientific Reports, 11(1), 1-10.
```

It is a constantly expanding project, and new functionalities are added as you read that text. Currently, I am implementing human mobility models - these are not functioning properly yet and are not covered in the documentation.

Current functionalities of HuMobi library cover:
* The basic class of TrajectoriesFrame which is used to load and store the mobility data. You will find that class in the
`structures` directory;
* Measures for individual and collective statistics in human mobility;
* Useful functions for data processing;
* Next-place and next time-bin prediction methods, including Markov Chains, deep-learning and shallow-learning models;
* Preprocessing methods for data aggregation and filtering;
* Other useful tools for data processing.

## Installing HuMobi

To install HuMobi and all dependencies simply use:

```
$ pip install HuMobi
```

Setting a proper working environment for this library can be tricky through the combination of specific libraries. Therefore, setting up virtualenv is recommended. Required dependencies for this library are:
* pandas >=1.1.5
* geopandas >=0.8.1
* tqdm >=4.59.0
* scipy >=1.5.2
* numpy >=0.19.5
* scikit-learn >=0.23.2
* biopython >=1.78
* shapely >=1.7.1
* numba >= 0.51.2
* tensorflow-gpu >= 2.4.1
* geofeather >=0.3.0

# Getting started

Below are the demos of various functionalities of this library. These cover most of the abilities of the current status of HuMobi. Note that this will be expanded in the future - including this documentation. For the method attributes and functions see the documentation in the HTML folder.

Some methods and functions need some time to execute. The `tqdm` library enables progress bars, which will estimate the remaining computation time.

## Data reading

Data loading, storing and reading are done within a special TrajectoriesFrame class. This is a pandas DataFrame-based data structure with MultiIndex, which consists of user id (upper level) and timestamp (lower level). Also, it contains a `geometry` column, identical to the GeoPandas GeoDataFrame geometry column.

First, let's import the necessary modules. We will import the trajectory module from humobi.structures and also pandas for the sake of this demo.
```
from humobi.structures import trajectory as tr
import pandas as pd
```

`TrajectoriesFrame` class is available in the trajectory module. `TrajectoriesFrame` is a smart class that will adjust to many data loading methods. For example, we can read a file given the path to it:
```
# READ FROM PATH
in_path = """brightkite_sample.tsv""" # PATH TO FILE
df = tr.TrajectoriesFrame(in_path, {'names': ['id', 'datetime', 'lat', 'lon', 'place'], 'delimiter': '\t',
                                    'crs': 4326})  # LOAD DATA
```
As you can see:
* the first positional argument is a file path.
* Apart from it, `**kwargs` of `pd.read_csv` function can be given. 
 
Additionally, `TrajectoriesFrame` accepts two metadata arguments:
* `crs` - Coordinate Reference System number according to EPSG classification 
* `geom_cols` - indicating two columns with coordinates.

Note that it is important to provide the `delimiter` keyword when it is other than a comma. 
Giving column names is useful, but `TrajectoriesFrame` will try to figure out which column has a timestamp and which has coordinates. However, to avoid errors provide columns with a timestamp as `time` or `datetime` and columns with geometry as `lat` and `lon`.

The first two lines of the file will look like that:
```
                                  id  ...                     geometry
user_id datetime                       ...                             
0       2010-10-17 01:48:53+00:00   0  ...  POINT (-104.99251 39.74765)
        2010-10-17 01:54:32+00:00   0  ...  POINT (-104.99251 39.74765)
```
> **_NOTE:_**  In debugging mode in some IDEs you can access a DataFrame view of TrajectoriesFrame through the `_base_class_view` protected attribute.

Another way to obtain a `TrajectoriesFrame` instance is to convert a standard DataFrame to it:
```
# CONVERT FROM DATAFRAME
df = pd.read_csv(in_path, names=['id', 'datetime', 'lat', 'lon', 'place'], delimiter='\t')  # LOAD DATA USING pd.read_csv
df = tr.TrajectoriesFrame(df)  # CONVERT TO TRAJECTORIESFRAME
```

Reading without column names will work, but is not recommended:
```
# CONVERT FROM DATAFRAME WITHOUT COLUMNS NAMES (NOT RECOMMENDED)
df = pd.read_csv(in_path, delimiter='\t', header=None)  # LOAD WITHOUT INFO ABOUT COLUMNS
df = tr.TrajectoriesFrame(df)
```

Also, you can read the file from geofeather file:
```
# READ FROM GEOFEATHER
df.from_geofeather("feather_path")
```

`TrajectoriesFrame` can be saved to CSV, geofeather or Shapefile using the below methods:
```
df.to_csv("csv_path")
df.to_geofeather("feather_path")
df.to_shapefile("shape_path")
```

`TrajectoriesFrame` overrides also the data spatial transformation method `to_crs`, which can be used to reproject the data. To do that, simply call:
```
df.to_crs(dest_crs = 3857, cur_crs = 4326)
```
This will reproject TrajectoriesFrame from `EPSG:4326` to `EPSG:3857`. If `cur_crs` is not given, TrajectoriesFrame will used `crs` metadata.

> **_NOTE:_**  `TrajectoriesFrame` is based on pandas DataFrame, hence it is possible to apply any pandas function - such as filtering and data selecting - on it.

## Data preprocessing

Raw movement data should be preprocessed to remove noise, remove unimportant stops and extract necessary information. HuMobi library offers methods for data preprocessing, analyses, and filtering. This process consists of two steps. First, noisy data is removed and stop locations are detected. In the second step, stop locations are aggregated into stay-regions and converted to movement sequence. For methodology details see publication:
```
Smolak, K., Siła-Nowicka, K., Delvenne, J. C., Wierzbiński, M., & Rohm, W. (2021). The impact of human mobility data scales and processing on movement predictability. Scientific Reports, 11(1), 1-10.
```

Data preprocessing tools are available in the `preprocessing` module. Some statistics and data compression methods are available in the `tools` module.
First, let's import necessary functions and read our data example saved with the `to_csv` method from the previous subsection of this readme.
```
from humobi.structures import trajectory as tr
from humobi.preprocessing.filters import stop_detection
from humobi.tools.user_statistics import *
from humobi.tools.processing import start_end

in_path = """converted_sample.csv"""
df_sel = tr.TrajectoriesFrame(in_path, {'crs': 4326})  # ALREADY CONVERTED - WITH GEOMETRY AND MULTIINDEX, SAVED TO CSV (SEE data_reading.py demo)
geom_cols = df_sel.geom_cols  # STORE GEOMETRY COLUMN
crs = df_sel.crs  # STORE CRS
```

### Data (users) selection

First, let's cover how to select particular movement trajectories to be able to filter the data later. `TrajectoriesFrame` offers the `uloc` method which allows you to select a user or users using their id. For example, let's select user of id `0`:
```
one_user = df_sel.uloc(0)
```

To get a list of all user ids, use `get_users()` method:
```
users_list = df_sel.get_users()  # LIST OF ALL IDS
```

Then, passing that list to `uloc` will result in selecting all users from the data (so the result will be unchanged):
```
many_users = df_sel.uloc(users_list)
```

You can use standard `loc` and `iloc` pandas commands, too.

### Users statistics

Many data selection methods are based on choosing users who have certain global statistics, like data completeness or trajectory duration. Module `tools.user_statistics` contains some metrics which can be used to calculate them. All results are returned as pandas `Series` with user id as index and statistics as values. Available statistics include:

#### Fraction of empty records
The fraction of empty records is expressed in a given temporal resolution. This is calculated globally - you can limit the timeframe using selection methods or use it together with `pd.rolling` to have a moving value.
```
frac = fraction_of_empty_records(df_sel, resolution='1H')  # FRACTION OF MISSING RECORDS
```
#### Total number of records
A total number of records:
```
count = count_records(df_sel)  # TOTAL NUMBER OF RECORDS
```
#### Total number of records calculated per time frame
A total number of records calculated per time frame:
```
count_per_time_frame = count_records_per_time_frame(df_sel, resolution='1D')  # TOTAL NUMBER OF RECORDS PER TIME FRAME
```

#### Trajectories duration
The total length of trajectories expressed in a time unit. `count_empty` determines whether empty records should be considered or excluded. If excluded, the value will be decreased by the number of empty records timeframes.
```
trajectories_duration = user_trajectories_duration(df_sel, resolution='1H', count_empty=False)  # TOTAL LENGTH OF TRAJECTORIES
```
#### The number of consecutive records
The highest number of consecutive records expressed in the given time unit.
```
consecutive = consecutive_record(df_sel, resolution='1H')
```

#### Filtering with statistics
Now, let's see how to use these statistics to filter some data. For example, we want to select only users with a fraction of empty records lower than 90%, whose trajectories are longer than 6 days and have at least 100 records of data. We will use sets intersection to find all the users that satisfy all these three requirements.
```
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

# REREAD STRUCTURE
df_sel = tr.TrajectoriesFrame(df_sel, {'crs': crs, 'geom_cols': geom_cols})
```

### Stop detection 

Stop detection algorithm is simple to use. `preprocessing.filters.stop_detection` function allows to quickly detect stay-points, by simply:
```
stops = stop_detection(df_sel, distance_condition=300, time_condition='10min')
```
For details on the algorithm see the mentioned publication. There are two parameters to adjust:
* `distance_condition` (here it is 300 metres) - always expressed in metres
* `time_condition` (here it is 10 minutes).

Function temporally converts data to `EPSG:3857`. Note that it uses multithreading, so having multiple cores helps.

`stop_detection` function adds a new boolean column `is_stop`. You can filter only stops using:
```
df_sel = stops[stops['is_stop'] == True]
```

It is also good to drop duplicates, as sometimes these may be created:
```
df_sel = df_sel.drop_duplicates()
```

Furthermore, to decrease data size, let's compress stops to a single row of data by adding `start` and `end` times of visits in these locations. To do that, call:
```
df_sel = start_end(df_sel)
```
Finally, resulting `TrajectoriesFrame` will look like this:
```
                                 id        lat         lon                                     place                     geometry  is_stop                      date                     start                       end
user_id datetime                                                                                                                                                                                                          
0       2009-05-29 00:04:23+00:00   0  39.759608 -104.984862          6346d66a3aa011de83f8003048c0801e  POINT (-104.98486 39.75961)     True 2009-05-29 00:04:23+00:00 2009-05-29 00:04:23+00:00 2009-05-29 02:29:20+00:00
        2009-05-30 02:12:30+00:00   0  39.890648 -105.068872          dd7cd3d264c2d063832db506fba8bf79  POINT (-105.06887 39.89065)     True 2009-05-30 02:12:30+00:00 2009-05-30 02:12:30+00:00 2009-05-30 07:28:16+00:00
```

### Data aggregation

After stay-point detection, data can be finally converted to movement sequences by, first, spatial (stay-regions detection), and then, temporal aggregation. 

Stay-regions detection can be done using various approaches, the most commonly used are the grid-based approach or clustering method. Also, there are two approaches to temporal aggregation: next time-bin and next place. Let's see how to convert our data to various movement sequences. First, let's import necessary functions.
```
from humobi.structures import trajectory as tr
from humobi.preprocessing.temporal_aggregation import TemporalAggregator
from humobi.preprocessing.spatial_aggregation import GridAggregation, ClusteringAggregator, LayerAggregator
from humobi.preprocessing.filters import next_location_sequence
```
#### Spatial aggregation

Spatial aggregation (stay-regions detection) should be done first. This process will add a new `labels` column which will identify unique stay-regions.

> **__NOTE:__** If you already have aggregated data and want to add unique labels per each geometry only, use `to_labels()` function from the `humobi.misc.utils` module.

`humobi.preprocessing.spatial_aggregation` module offers following spatial aggregation classes:
* `GridAggregator`, 
* `ClusteringAggregator`, 
* `LayerAggregator` 

To perform aggregation, an aggregator class has to be defined first. When an aggregator is created, the data and arguments controlling aggregation behaviour are passed first. After that, the `aggregate` method can be called to perform data aggregation.

##### Grid Aggregator

`GridAggregator` is a quick data aggregation to a regular grid of defined resolution. There are some implemented behaviours. For example, you can pass only the `resolution` argument, and the grid will fit into the data extent.
```
gird_resolution = 1000  # DEFINE SPATIAL UNIT (GRID)
grid_agg = GridAggregation(gird_resolution)  # DEFINE GRID AGGREGATION ALGORITHM
df_sel_grid = grid_agg.aggregate(df_sel, parallel=False)  # CALL AGGREGATION
```
* When you want to set the grid extent yourself, you can pass `x_min`, `x_max`, `y_min`, `y_max` parameters to set the extent of the aggregation grid. 
* You can pass an `origin` parameter to tell whether the grid should be centred at the data.
* `parallel` parameter of the `aggregate()` method calls multithread processing, but this is not necessarily faster than the single-core method, due to its overheads.

##### Clustering Aggregator

`ClusteringAggregator` allows you to pass any scikit-learn clustering algorithm to perform clusterisation of the stay-points. In the below example we use the `DBSCAN` class to perform clustering:
```
from sklearn.cluster import DBSCAN
eps = 300  # DEFINE SPATIAL UNIT
min_pts = 2  # OTHER HYPERPARAMETERS
clust_agg = ClusteringAggregator(DBSCAN, **{"eps": eps, "min_samples": min_pts})  # DEFINE SPATIAL AGGREGATION ALGORITHM
df_sel_dbscan = clust_agg.aggregate(df_sel)  # SPATIAL AGGREGATION CALL
```
As you see, all the arguments for the clustering method can be passed as `**kwargs`. This class uses multithreading implemented within the scikit-learn library.

##### Layer Aggregator

The third class is `LayerAggregator`. This class uses an external file to perform aggregation. Its functionality is based on the GeoPandas function of spatial join. To perform it, just call:
```
layer_agg = LayerAggregator('path_to_outer_layer',**kwargs)
df_sel_layer = layer_agg.aggregate(df_sel)
```

#### Temporal aggregation

Temporal aggregation functionalities are available in the `humobi.preprocessing.temporal_aggregation` module, which contains the `TemporalAggregator` class. There are two approaches to temporal aggregation: next time-bin and next place.

##### Next time-bin
The next time-bin approach converts sequences of visited locations into evenly spaced time-bins. To perform the next time-bin aggregation, simply instantiate TemporalAggregator class and pass time unit which will be used to perform aggregation:
```
time_unit = '1H'  # DEFINE TEMPORAL UNIT
time_agg = TemporalAggregator(time_unit)  # DEFINE TEMPORAL AGGREGATION ALGORITHM
````
In the above example, we chose time-bins to have an hourly resolution. Now we can call the aggregation of our data:
```
df_sel_dbscan_time = time_agg.aggregate(df_sel_dbscan, parallel=True) 
```
The above line of code will perform the time-bin aggregation according to the methodology presented in the literature. Three cases may occur:
* No data was found for the time-bin -> In that case, the time-bin will be empty
* There was more than one stay-region visited during a time-bin -> Stay-region, where the user spent more time, is selected
* There was more than one stay-region visited during a time-bin and the user spent an identical amount of time in them -> Stay-region, where the user spent more time in the past, is selected

* Temporal aggregation can be run using the `parallel` setting, which is a bit faster than its single-core variant. 
* `aggregate()` method has a `fill_method` argument, which can be set to `ffill` or `bfill` to fill missing data or a `drop_empty` argument, which can be used to remove missing time-bins.

Temporal aggregation is computationally heavy and can take some time.

> **_NOTE:_** Time-bins always start at midnight.

##### Next place

In the next place approach, all the consecutive records of visits in the same stay-region are removed. This can be done using `next_location_sequence` function from the `humobi.preprocessing.filters` module.
```
df_sel_time_seq = next_location_sequence(df_sel_dbscan_time)  # CREATE NEXT LOCATION SEQUENCES
```

## Metrics

Once processed, various metrics can be calculated on movement sequences. We divide them into individual and collective. Individual metrics are calculated separately for each id in the `TrajectoriesFrame`. Collective metrics are presented in the form of distributions or are referred to as stay-regions.

First, let's import all the metrics:
```
from humobi.structures import trajectory as tr
from humobi.measures.individual import *
from humobi.measures.collective import *
```

We assume our processed data are stored under the `df_sel_dbscan_time` variable.

### Individual metrics

#### Number of distinct locations

This metric calculates the number of distinct locations visited by each individual. To calculate it, call `num_of_distinct_locations()` function from `humobi.measures.individual` module.
```
distinct_total = num_of_distinct_locations(df_sel_dbscan_time)
```

#### Visitation frequency

This metric calculates the frequency of visits in each stay-region visited by users. Execute:
```
vfreq = visitation_frequency(df_sel_dbscan_time)
```

#### Number of distinct locations over time

This is a variant of the number of distinct locations metric and calculates the number of distinct locations visited from the start of the movement trajectory at each time step. This function requires two additional parameters. 
* `resolution` determines the size of a time step. 
* `reaggregate` is a boolean parameter, which will run TemporalAggregator to convert data into a new time-bin size if needed. 

Execution:
```
distinct_over_time = distinct_locations_over_time(df_sel_dbscan_time, resolution='1H', reaggregate=False)
```

#### Jump lengths
This function calculates the length of all trips between locations in the movement sequence.
```
jump = jump_lengths(df_sel_dbscan_time)
```

#### Nonzero trips
This function calculates the number of all trips (which covered distance > 0)
```
trips = nonzero_trips(df_sel_dbscan_time)
```

#### Self-transtition
This function calculates the number of situations when the user stayed in the same location for the next time-bin (in the next place it will always be equal to 0).
```
st = self_transitions(df_sel_dbscan_time)
```

#### Waiting times
This function calculates waiting times for each transition in `TrajectoriesFrame`. 
* This function requires `time_unit` to be specified, which will control the unit in which waiting time will be expressed.
```
wt = waiting_times(df_sel_dbscan_time, time_unit = 'h')
```

#### Center of mass
Calculates a centre of mass for each user's trajectory.
```
mc = center_of_mass(df_sel_dbscan_time)
```

#### Radius of gyration
Calculates radii of gyration for each user. 
* Optionally `time_evolution` argument can be used to express this metric evolution in time.
```
rog = radius_of_gyration(df_sel_dbscan_time, time_evolution=False)
rog_time = radius_of_gyration(df_sel_dbscan_time, time_evolution=True)
```

#### Mean square displacement
Calculates mean square displacements (MSD) for each user. 
* Optionally `time_evolution` argument can be used to express this metric evolution in time.
```
msd = mean_square_displacement(df_sel_dbscan_time, time_evolution=False)
msd_time = mean_square_displacement(df_sel_dbscan_time, time_evolution=True)
```
* The `from_center` argument can be used to calculate MSD in reference to the centre of trajectory mass (if False it is calculated from the first point). 
* `reference_locs` can be passed to determine custom reference points for each person. This has to be a series with users' id as index and point geometry as values.

#### Return time
Calculates return times for each unique location in each user's trajectory. 
* `time_unit` specifies the unit in which return times will be expressed.
```
rt = return_time(df_sel_dbscan_time, time_unit = 'H')
```
Optionally this metric can be calculated in relation to places, which will express how long it takes any person to return to that location on average. This will produce a DataFrame with the count of returns and mean time to return.
```
rt_place = return_time(df_sel_dbscan_time, by_place=True)
```

#### Random entropy and predictability
Calculates random entropy for each user in TrajectoriesFrame using equation defined in 
`Song, C., Qu, Z., Blumm, N., & Barabási, A. L. (2010). Limits of predictability in human mobility. Science, 327(5968), 1018–1021. https://doi.org/10.1126/science.1177170`. Similarly, predictability is calculated using entropy and Fano's inequality as presented in the above paper.
```
ran_ent = random_entropy(df_sel_dbscan_time)
random_pred = random_predictability(df_sel_dbscan_time)
```
> **__NOTE:__** `random_predictability` returns entropy and predictability in a DataFrame.

#### Uncorrelated entropy and predictability
Calculates uncorrelated entropy for each user in TrajectoriesFrame using equation defined in 
`Song, C., Qu, Z., Blumm, N., & Barabási, A. L. (2010). Limits of predictability in human mobility. Science, 327(5968), 1018–1021. https://doi.org/10.1126/science.1177170`. Similarly, predictability is calculated using entropy and Fano's inequality as presented in the above paper.
```
unc_ent = unc_entropy(df_sel_dbscan_time)
unc_pred = unc_predictability(df_sel_dbscan_time)
```
> **__NOTE:__** `unc_predictability` returns entropy and predictability in a DataFrame.

#### Real entropy and predictability
Calculates real entropy for each user in TrajectoriesFrame using the Lempel-Ziv compression algorithm, using approach defined in
`Song, C., Qu, Z., Blumm, N., & Barabási, A. L. (2010). Limits of predictability in human mobility. Science, 327(5968), 1018–1021. https://doi.org/10.1126/science.1177170`, and corrected using findings from
`Xu, P., Yin, L., Yue, Z., & Zhou, T. (2019). On predictability of time series. Physica A: Statistical Mechanics and Its Applications, 523, 345–351. https://doi.org/10.1016/j.physa.2019.02.006`
and
`Smolak, K., Siła-Nowicka, K., Delvenne, J. C., Wierzbiński, M., & Rohm, W. (2021). The impact of human mobility data scales and processing on movement predictability. Scientific Reports, 11(1), 1–10. https://doi.org/10.1038/s41598-021-94102-x`
Additionally, when fraction of missing data is higher than 15%, entropy is estimated using an approach from `Ikanovic, E. L., & Mollgaard, A. (2017). An alternative approach to the limits of predictability in human mobility. EPJ Data Science, 6(1). https://doi.org/10.1140/epjds/s13688-017-0107-7`.

Predictability is calculated using entropy and Fano's inequality as presented in the Song et al. (2010) paper.

> **__NOTE:__** Real entropy __cannot__ be calculated when fraction of missing data is >90%.

> **__NOTE:__** These function are using GPU to perform calculations. Be sure to have CUDA configured on your machine. CPU variant is not accessible, because calculations on the CPU take an unreasonable long time to execute.
```
real_ent = real_entropy(df_sel_dbscan_time)
real_pred = real_predictability(df_sel_dbscan_time)
```
> **__NOTE:__** `real_predictability` returns entropy and predictability in a DataFrame.

#### Stationarity
Calculates the stationarity according to Teixeira et al. (2019) as the average stay length in the location. See `Teixeira, D., Almeida, J., Viana, A. C., Teixeira, D., Almeida, J., Carneiro, A., … Viana, A. C. (2021). Understanding routine impact on the predictability estimation of human mobility To cite this version : HAL Id : hal-03128624 Understanding routine impact on the predictability estimation of human mobility.` for details.

```
stat = stationarity(df_sel_dbscan_time)
```

#### Regularity
Calculates the regularity according to Teixeira et al. (2019) as the ratio of sequence length and the number of unique symbols. See `Teixeira, D., Almeida, J., Viana, A. C., Teixeira, D., Almeida, J., Carneiro, A., … Viana, A. C. (2021). Understanding routine impact on the predictability estimation of human mobility To cite this version : HAL Id : hal-03128624 Understanding routine impact on the predictability estimation of human mobility.` for details.

```
regul = regularity(df_sel_dbscan_time)
```

### Collective metrics

#### Distribution of travelling distances
Calculates the distribution of travelling distances for each user. Two mutually exclusive parameters can be defined:
* `bin_size` - determines the size of a bin in an output histogram,
* `n_bins` - determines the number of bins in an output histogram. (default = 20)
```
dist = dist_travelling_distance(df_sel_dbscan_time)
```

#### Pairwise comparison of flows
Calculates the number of flows for each aggregation cell. 
* Using `flows_type` - `all` flows, only `incoming`, or only `outgoing` flows can be counted.
```
pairwise_flows = flows(df_sel_dbscan_time, flows_type='all')
```

## Data generation routines

Generating synthetic data might be useful to verify algorithms and assumptions on sequences of known statistical properties. There are few generation routines available in the `humobi.misc.generators` module. The output is a DataFrame with `labels` columns, which identify unique locations:
```
                             labels
user_id datetime                                                      
0       1970-01-01 00:00:00       0 
        1970-01-01 01:00:00       0
```

Each routine has available parameters:
* `users` - which is the number of unique sequences to generate
* `places` - which is the size of vocabulary for generation
Fixed values or a list of values can be passed to these arguments. If the latter is used, each sequence will be generated using a randomly picked value from the list. Some routines have more parameters. See details below.

To import module:
```
from humobi.misc.generators import *
```

### Random sequences


Random sequences, where each symbol is randomly generated from the available vocabulary. Additional `length` parameters can be determined. It can be a single value or a list of values to randomly pick from.
```
random_seq = random_sequences_generator(users=10, places=10, length=100)
```

### Deterministic sequences

Deterministic sequences follow a series of incrementing symbols up to the size of the vocabulary. At the end of vocabulary, series are repeated. For example, when vocabulary size is 4, then sequence will follow routine `[0, 1, 2, 3, 0, 1, 2, 3, 1, ...]`. Additional `repeats` parameters can be determined. It can be a single value or a list of values to randomly pick from.
```
deter_seq = deterministic_sequences_generator(users=10, places=10, repeats=10)
```

### Markovian sequences

Markovian sequences follow a deterministic sequence, but at each step with probability `prob`, a random symbol is inserted.
```
markovian_seq = markovian_sequences_generator(users=10, places=10, length=500, prob=.3)
```
### Exploratory sequences
Exploratory sequences generate a sequence of unique, non-repeating symbols.

```
ex_seq = exploratory_sequences_generator(users=10, places=10)
```
### Self-transition sequences
Self-transitions sequences are similar to deterministic sequences, but each symbol is repeated multiple times before moving to the next one. The number of self-transitions repeating after each other is determined by the number of symbols and the length of the sequence.
```
st_seq = self_transitions_sequences_generator(users=10, places=10, length=100)
```

### Non-stationary sequences
Non-stationary sequences generate symbols using `states`, where each state has its routine of symbols generation. Probabilities for each state are assigned randomly at the beginning of the process.
```
non_st_seq = non_stationary_sequences_generator(users=10, places=10, states=5, length=100)
```
## Next location predictions

Finally, let's do some predictions. There are multiple algorithms available, including:
* Naive approaches
* Markov chains
* Shallow learning classification algorithms
* Deep learning networks

First, let's import the necessary modules:
```
from humobi.predictors.wrapper import *
from humobi.predictors.deep import *
```
> **__NOTE:__** This module is under development, there are some still inconsistencies in the structure. This will be adjusted in the future, to facilitate pipeline creation.

### Data split

To perform predictions, we need to split data into training and testing sets.  We will use the `Splitter` class from the `humobi.predictors.wrapper` module. This class allows splitting the whole TrajectoriesFrame at once by splitting each users' trajectory in the determined ratio (determined by the `split_ratio` argument which expresses the test set size). 
For example, let's split generated Markovian sequences:
```
data_splitter = Splitter(markovian_seq, split_ratio=.2)
```
Split data will be accessible as `split` instance attributes.
* `split.data` gives access to the input data
* `split.cv_data` yields a list of training-validation sets pairs. An additional parameter `n_splits` can be passed to the `Splitter`, which will generate more pairs of training-validation datasets pairs using the KFold approach. Each pair consists of training set features (index 0), training set labels (index 1), validation set features (index 2), and validation set labels (index 3).
* `split.test_frame_X` gives access to the test set features
* `split.text_frame_Y` gives access to the test set labels

* a `horizon` parameter can be set on the `Splitter` class. If `horizon > 1` all the generated datasets of features (X sets) will contain additional columns with all previous time steps, up to the horizon length. Moreover, training/validation and testing sets will overlap with the `horizon` size.

Let's assign our datasets to some variables:
```
test_frame_X = data_splitter.test_frame_X
test_frame_Y = data_splitter.test_frame_Y
cv_data = data_splitter.cv_data
```

Once data is split, now we can make predictions.

### TopLoc

The TopLoc algorithm is a naive algorithm adapted from `Cuttone, A., Lehmann, S., & González, M. C. (2018). Understanding predictability and exploration in human mobility. EPJ Data Science, 7(1). https://doi.org/10.1140/epjds/s13688-017-0129-1`. When doing prediction, it assumes that the next visited place is the most visited place in the training set. `TopLoc()` class is available in the `humobi.predictors.wrapper` module. To perform prediction on the datasets call:
```
TopLoc(train_data=cv_data, test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y]).predict()
```
Calling `predict()` will return predictions (DataFrame, where the first columns are test set labels and the second column are predictions) and the accuracy score.

### MarkovChain

N-th Markov Chain can be called through wrapper function `markov_wrapper`. This wrapper accepts the whole dataset at once and performs a data split inside.  Test-train ratio is defined by the `test_size` parameter. The order of the Markov Chain is defined by the `state_size` parameter. For example, to call 2nd-order Markov Chain, execute the following line:
```
MC2 = markov_wrapper(markovian_seq, test_size=.2, state_size=2)
```
Additionally, this function accepts two arguments: `update` - if True, with each predicted symbol chain is rebuilt, and `online` - which if True, allows the algorithm to see the last `state_size` symbols from the test set when predicting. This function returns the prediction and the accuracy score.

### Shallow learning methods

It is possible to use scikit-learn library classification methods to perform predictions. `humobi.predictors.wrapper` module allows to use `SKLearnPred()` class to apply any algorithm to perform the next location prediction. To do that, first, let's import a prediction algorithm:
```
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier
```
After we instantiate the `SKLearnPred` class in the following way:
```
predic = SKLearnPred(algorithm=clf, training_data=split.cv_data, test_data=[split.test_frame_X, split.test_frame_Y], param_dist={'n_estimators': [x for x in range(500, 5000, 500)], 'max_depth': [None, 2, 4, 6, 8]}, search_size=5, cv_size=5, parallel=False)
```
* The algorithm has to be passed as an `algorithm` argument. 
* `training_data` and `test_data` have to be passed from the `Splitter()` class.
* `param_dist` allows passing a dictionary of candidate hyperparameters of prediction algorithms. These will be tested to find the best solution during RandomizedSearch with cross-validation. 
* The number of cross-validations is determined by the `cv_size` parameter.
* The number of tested combinations is determined by the `search_size` parameter. As of now `parallel` computing is not possible.

To perform predictions, first, we need to train prediction algorithms. Calling
```
predic.learn()
```
will train a set of algorithms, each for each trajectory in the dataset. Then calling
```
predic.test()
```
will perform prediction on the test set. Accuracy can be accessed through `predic.scores` attributes, predictions can be accessed through `predic.predictions` attribute.

### Deep learning methods

As of now, the HuMobi library offers two network architectures for mobility predictions. This will be extended in the future. These are single- and double-layer Gated Recurrent Unit (GRU) networks with embedding and dropout. These are accessible in the `humobi.predictors.deep` module.
To perform predictions with a deep learning network, first, call the `DeepPred` class. This class accepts the following arguments at initialisation:
* `model` - to use single-layer GRU call `'GRU'`, to use double-layer GRU call `'GRU2'`
* `data` - TrajectoriesFrame (not from Splitter)
* `test_size` - size of the test set
* `folds` - number of folds in cross-validation to use
* `batch_size` - the size of batches for the network.
* `embedding_dim` - the dimensionality of embedding vector fit at the first layer of the network
* `rnn_units` - the number of neurons in the GRU layers

Example of usage:
```
GRU = DeepPred("GRU", markovian_seq, test_size=.2, folds=5, window_size=5, batch_size=1, embedding_dim=512, rnn_units=1024)
```
To perform prediction, call:
```
GRU.learn_predict()
```
To access predictions call `GRU.predictions`, to access accuracy scores call `GRU.scores`.

## Paper: Explaining human mobility predictions through pattern matching algorithm

Publication `Explaining human mobility predictions through a pattern matching algorithm` by `Kamil Smolak, Witold Rohm, and Katarzyna Siła-Nowicka` presents and evaluate five new metrics which measure the actual predictability of mobility data and explain the variability in accuracy of the actual predictions. These are:
* Dense Repeatability (DR)
* Sparse Repeatability (SR)
* Equally Sparse Repeatability (ESR)
* Global Alignment (GA)
* Iterative Global Alignment (IGA)

These metrics are implemented within the HuMobi library and are available in the `humobi.measures.individual` module. Apart from the above metrics, all the works performed in this paper were done using this library. This includes data filtration and preprocessing (see [Data preprocessing](#Data-preprocessing)) some of the calculated metrics (especially real predictability, stationarity, and regularity, see [Metrics](#Metrics) for details). Predictions and their accuracy were done as shown in the [Next location predictions](#Next-location-predictions) section.

These metrics are based on the training and test sets. These can be achieved from `Splitter()`:
```
X = pd.concat([split.cv_data[0][0], split.cv_data[0][2]])
Y = split.test_frame_X
```

Any of the above metrics can be simply calculated by passing `train_frame` and `test_frame` arguments:
```
DR = repeatability_dense(train_frame=X, test_frame=Y)
SR = repeatability_sparse(train_frame=X, test_frame=Y)
ESR = repeatability_equally_sparse(train_frame=X, test_frame=Y)
GA = global_alignment(train_frame=X, test_frame=Y)
IGA = iterative_global_alignment(train_frame=X, test_frame=Y)
```
This will return pd.Series with values calculated for each user in the data.

## Known issues

One of the known issues is related to crs metadata setting. It may happen, that giving the `crs` argument to `TrajectoriesFrame` does not assign it to the dataset. This is a well known GeoPandas library and Anaconda issue. Proper setting of the proj4 library solves this problem.

Based on Dorregaray's answer from here: [StackOverflow question](#https://stackoverflow.com/questions/55390492/runtimeerror-bno-arguments-in-initialization-list/58009620#58009620), a solution is to edit the path in the `datadir.py` file located in `...\Anaconda3\Lib\site-packages\pyproj`. It should renamed to `.../Anaconda3/Library/share`.
