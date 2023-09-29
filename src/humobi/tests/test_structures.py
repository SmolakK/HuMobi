import sys

sys.path.append('..')
import src.humobi.structures.trajectory as traj
import pandas as pd
import geopandas as gpd
import numpy as np
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal


class TestStructures:

	def test_read_from_file(self):
		# With header test
		file_path = r"sample_data_header.csv"
		result = traj.TrajectoriesFrame(file_path, {'crs': 27700})
		data = \
			[
				[5, '2013-09-29 08:00:00', 0, 311230.97130721284, 686330.4551903725, 0, '2013-09-29 08:46:53',
				 '2013-09-29 09:05:58'],
				[5, '2013-09-29 09:00:00', 2.0, 311226.8336536617, 686308.4508384819, 1.0, '2013-09-29 09:34:06',
				 '2013-09-29 11:44:52'],
				[7, '2014-10-03 02:00:00'],
				[7, '2014-10-03 10:00:00', 13.0, 323453.54817847826, 672132.333636006, 1.0, '2013-10-03 06:44:11',
				 '2013-10-03 12:18:58']
			]
		data = pd.DataFrame(data, columns=['user_id', 'datetime', 'temp', 'lat', 'lon', 'labels', 'start', 'end'])
		data['datetime'] = pd.to_datetime(data['datetime'])
		expected = gpd.GeoDataFrame(data=data, geometry=gpd.points_from_xy(*[data[x] for x in ['lat', 'lon']]),
		                            crs=27700)
		result = gpd.GeoDataFrame(result).reset_index()
		assert_geodataframe_equal(result.dropna(), expected.dropna())
		# No header test
		file_path = r"sample_data_noheader.csv"
		result = traj.TrajectoriesFrame(file_path, {'crs': 27700,
		                                            'names': ['user_id', 'datetime', 'temp', 'lat', 'lon', 'labels',
		                                                      'start', 'end', 'geometry']})
		data = \
			[
				[5, '2013-09-29 08:00:00', 0, 311230.97130721284, 686330.4551903725, 0, '2013-09-29 08:46:53',
				 '2013-09-29 09:05:58'],
				[5, '2013-09-29 09:00:00', 2.0, 311226.8336536617, 686308.4508384819, 1.0, '2013-09-29 09:34:06',
				 '2013-09-29 11:44:52'],
				[7, '2014-10-03 02:00:00'],
				[7, '2014-10-03 10:00:00', 13.0, 323453.54817847826, 672132.333636006, 1.0, '2013-10-03 06:44:11',
				 '2013-10-03 12:18:58']
			]
		data = pd.DataFrame(data, columns=['user_id', 'datetime', 'temp', 'lat', 'lon', 'labels', 'start', 'end'])
		data['datetime'] = pd.to_datetime(data['datetime'])
		expected = gpd.GeoDataFrame(data=data, geometry=gpd.points_from_xy(*[data[x] for x in ['lat', 'lon']]),
		                            crs=27700)
		result = gpd.GeoDataFrame(result).reset_index()
		assert_geodataframe_equal(result.dropna(), expected.dropna())
		# No geometry test
		file_path = r'sample_date_nogeom_noheader.csv'
		result = traj.TrajectoriesFrame(file_path, {'crs': 27700,
		                                            'names': ['user_id', 'datetime', 'temp', 'lat', 'lon', 'labels',
		                                                      'start', 'end', 'geometry']})
		data = \
			[
				[5, '2013-09-29 08:00:00', 0, 311230.97130721284, 686330.4551903725, 0, '2013-09-29 08:46:53',
				 '2013-09-29 09:05:58'],
				[5, '2013-09-29 09:00:00', 2.0, 311226.8336536617, 686308.4508384819, 1.0, '2013-09-29 09:34:06',
				 '2013-09-29 11:44:52'],
				[7, '2014-10-03 02:00:00'],
				[7, '2014-10-03 10:00:00', 13.0, 323453.54817847826, 672132.333636006, 1.0, '2013-10-03 06:44:11',
				 '2013-10-03 12:18:58']
			]
		data = pd.DataFrame(data, columns=['user_id', 'datetime', 'temp', 'lat', 'lon', 'labels', 'start', 'end'])
		data['datetime'] = pd.to_datetime(data['datetime'])
		expected = gpd.GeoDataFrame(data=data, geometry=gpd.points_from_xy(*[data[x] for x in ['lat', 'lon']]),
		                            crs=27700)
		result = gpd.GeoDataFrame(result).reset_index()
		assert_geodataframe_equal(result.dropna(), expected.dropna())
