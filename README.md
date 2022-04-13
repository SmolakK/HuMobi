# HuMobi
 
 ## Table of contents
* [General info](#General-info)
* [Installing HuMobi](#Installing-HuMobi)
* [Data reading](#Data-reading)
* [Data preprocessing](#Data-preprocessing)
 
## General Info
This is the HuMobi library. It is a dedicated Python library for human mobility data processing, which mostly extends Pandas
DataFrames and Geopandas GeoDataFrames to facilitate operating on a very specific data structure of individual mobility trajectories. Below you will find info how to install a HuMobi library at your computer and some demos covering most of library functionalities.

This library is mainly devoted to processing individual mobility trajectories and focuses on human mobility prediction and modelling. Initially it was implemented during PhD studies by Kamil Smolak, a PhD candidate at the Wrocław University of Environmental and Life Sciences.

If you use this library please cite below work:
```
Smolak, K., Siła-Nowicka, K., Delvenne, J. C., Wierzbiński, M., & Rohm, W. (2021). The impact of human mobility data scales and processing on movement predictability. Scientific Reports, 11(1), 1-10.
```

It is a constantly expanding project, and new functionalities are added as you read that text. Currently, I am implementing human mobility models - these are not functioning properly yet and are not covered in the documentation.

Current functionalities of HuMobi library cover:
* The basic class of TrajectoriesFrame - used to load and store the mobility data. You will find that class in the
structures directory.
* Measures of individual and collective statistics for human mobility.
* Useful functions for data processing.
* Next-place and next time-bin prediction methods, including Markov Chains, deep-learning and shallow-learning models.
* Preprocessing methods for data aggregation and filtering.
* Other useful tools for data processing.

## Installing HuMobi

To install HuMobi and all dependencies simply use:

```
$ pip install HuMobi
```

Setting properly working environment for this library can be tricky through the combination of specific libraries. Therefore, setting up virtualenv is recommended. Required dependencies for this library are:
* pandas
* geopandas
* tqdm
* scipy
* numpy
* scikit-learn
* Biopython
* shapely
* numba
* tensorflow-gpu
* geofeather

# Getting started

Below are demos of various funcionalities of this library. These cover majority of this library abilites. Note that this will be expanded in the future - including this documentation. For method attributes and functions see documentation in html folder.

## Data reading

Data loading, storing and reading is done within a special TrajectoriesFrame class. This is a pandas DataFrame-based data structure with MultiIndex, which consists of user id (upper level) and timestamp (lower level). Also, it will contain a `geometry` column, identically to GeoPandas GeoDataFrame geometry column.

First, let's import necessary modules. We will import trajectory module from humobi.structures and also pandas for the sake of this demo.
```
from humobi.structures import trajectory as tr
import pandas as pd
```

TrajectoriesFrame class is available in trajectory module. `TrajectoriesFrame` is a smart class which will adjust to many data loading methods. For example, we can read file given the path to it:
```
# READ FROM PATH
in_path = """brightkite_sample.tsv""" # PATH TO FILE
df = tr.TrajectoriesFrame(in_path, {'names': ['id', 'datetime', 'lat', 'lon', 'place'], 'delimiter': '\t',
                                    'crs': 4326})  # LOAD DATA
```
As you can see, the first positional argument is a file path. Apart from it, kwargs of pd.read_csv function can be given. Additionally, `TrajectoriesFrame` accepts two metadata arguments `crs` - Corrdinate Reference System number according to EPSG classification and `geom_cols` - indicating two columns with coordinates.
Note that it is important to provide `delimiter` keyword, when it is other than comma. Giving column names is useful, but `TrajectoriesFrame` will try to figure out which column has timestamp and which has coordinates. However, to avoid errors provide columns with timestamp as `time` or `datetime` and columns with geometry as `lat` and `lon`.
The first line of file will look like that:
```
                                  id  ...                     geometry
user_id datetime                       ...                             
0       2010-10-17 01:48:53+00:00   0  ...  POINT (-104.99251 39.74765)
```
> **_NOTE:_**  In debugging mode in some IDEs you can access a DataFrame view of TrajectoriesFrame through `_base_class_view` protected attribute.

Another way to obtain a `TrajectoriesFrame` instance is to convert standard DataFrame to it:
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

TrajectoriesFrame can be saved to csv, geofeather or Shapefile using below methods:
```
df.to_csv("csv_path")
df.to_geofeather("feather_path")
df.to_shapefile("shape_path")
```

## Data preprocessing
