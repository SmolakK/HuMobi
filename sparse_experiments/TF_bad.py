from humobi.structures import trajectory as tr
import pandas as pd
pth = """C:\\Users\\kamil\\Downloads\\boars.csv"""
import geopandas as gpd
df = pd.read_csv(pth,index_col=[0,1])
df = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(*[df[x] for x in ['lat','lon']]))
df = df.set_crs("epsg:4326")
df._crs = 4326
print(df._crs)
tr.TrajectoriesFrame(df, {'crs': 4326})