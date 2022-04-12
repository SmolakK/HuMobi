from humobi.structures import trajectory as tr
import pandas as pd

# READ FROM PATH
in_path = """brightkite_sample.tsv"""
df = tr.TrajectoriesFrame(in_path, {'names': ['id', 'datetime', 'lat', 'lon', 'place'], 'delimiter': '\t',
                                    'crs': 4326})  # LOAD DATA
geom_cols = df.geom_cols  # ASSIGN GEOMETRY COLUMN
crs = df.geom_cols  # ASSIGN CRS
df.to_csv("converted_sample.csv")  # You can use Pandas and Geopandas methods to read and save data

# CONVERT FROM DATAFRAME

df = pd.read_csv(in_path, names=['id', 'datetime', 'lat', 'lon', 'place'], delimiter='\t')  # LOAD DATA
df = tr.TrajectoriesFrame(df)  # CONVERT
geom_cols = df.geom_cols  # ASSIGN GEOMETRY COLUMN
crs = df.geom_cols  # ASSIGN CRS

# CONVERT FROM DATAFRAME WITHOUT COLUMNS NAMES (NOT RECOMMENDED)
df = pd.read_csv(in_path, delimiter='\t', header=None)  # LOAD WITHOUT INFO ABOUT COLUMNS
df = tr.TrajectoriesFrame(df)
geom_cols = df.geom_cols  # ASSIGN GEOMETRY COLUMN
crs = df.geom_cols  # ASSIGN CRS

# SAVE TO GEOFEATHER (rename columns if not str)
df.to_geofeather("feather_path")

# READ FROM GEOFEATHER
df.from_geofeather("feather_path")

# SAVE TO SHAPEFILE
df.to_shapefile("shape")
