import pandas as pd
import geopandas as gpd
import numpy as np
import re
import geofeather as gf
import warnings


def infer_datetime(trajectories_frame):
	"""
	This function infers the timestamp column index.

	Args:
		trajectories_frame (TrajectoriesFrame): TrajectoriesFrame class object

	Returns:
		date column index and time column index or datetime column index
	"""
	single_row = trajectories_frame.head(1)
	date_col = None
	time_col = None
	for name, val in single_row.iteritems():
		val = val.iloc[0]
		if re.match(r".+(\/).+(\/).+", str(val)):
			date_col = name
		elif re.match(r".+(-).+(-).+", str(val)):
			date_col = name
		if re.match(r".{1,2}(:).{1,2}(:).{1,2}", str(val)):
			time_col = name

	if date_col == time_col:
		return date_col
	elif isinstance(date_col, str) and time_col is None:
		return date_col
	else:
		return date_col, time_col


def infer_geometry(trajectories_frame):
	"""
	This function infers the index of column with geometry
	
	Args:
		trajectories_frame (TrajectoriesFrame): TrajectoriesFrame class object
	
	Returns:
		geometry column index
	"""
	latitude_column = None
	longitude_column = None
	if trajectories_frame.columns.dtype is 'str' or trajectories_frame.columns.is_object():
		try:
			latitude_column = [x for x in trajectories_frame.columns if 'lat' in x.lower()][0]
		except IndexError:
			pass
		try:
			longitude_column = [x for x in trajectories_frame.columns if 'lon' in x.lower()][0]
		except IndexError:
			pass
	if not latitude_column or not longitude_column:
		try:
			latitude_column, longitude_column = trajectories_frame.columns[
				trajectories_frame.dtypes == 'float64'].values
			warnings.warn("TrajectoriesFrame: Selecting first two columns of float")
		except:
			trajectories_frame['lat'] = 0
			trajectories_frame['lon'] = 0
			latitude_column = 'lat'
			longitude_column = 'lon'
			warnings.warn("TrajectoriesFrame: No coordinates, adding empty floats")
	return latitude_column, longitude_column


def infer_id(trajectories_frame):
	"""
	This function infers the index of column with user identifier
	:param trajectories_frame: TrajectoriesFrame class object
	:return: identifier column index
	"""
	if trajectories_frame.columns.dtype is 'str' or trajectories_frame.columns.is_object():
		try:
			id_col = [x for x in trajectories_frame.columns if 'id' in x.lower()][0]
			return id_col
		except:
			raise KeyError("TrajectoriesFrame has to contain ID column (with id in the name)!")


class TrajectoriesFrame(gpd.GeoDataFrame):
	"""
	This is a basic class of TrajectoriesFrame. It stores user traces with MultiIndex - sorted by ID, and by timestamp.
	This class inherits from GeoPandas DataFrame, therefore has geographical properties.
	"""

	_metadata = ['_crs', '_geom_cols', ]

	@property
	def _constructor(self):
		def _c(data):
			return TrajectoriesFrame(data).__finalize__(self)

		return _c

	def __init__(self, data=None, params={}):
		"""
		Creates TrajectoriesFrame from input - can be DataFrame or file path
		:param data: DataFrame or file path
		:param params: dictionary of parameters accepted by DataFrame class
		"""
		try:
			self._crs = data.crs
		except AttributeError:
			self._crs = params.pop('crs', None)

		if isinstance(data, str):
			data = pd.read_csv(data, **params)

		columns_user_defined = params.pop('columns', None)
		self._geom_cols = params.pop('geom_cols', None)

		if isinstance(data, pd.DataFrame):
			if not data.index.nlevels > 1:
				if len(data.select_dtypes(include=[np.datetime64])) > 1:
					if 'datetime' in data.columns:
						date_cols = 'datetime'
					else:
						date_cols = infer_datetime(data)
					if isinstance(date_cols, str):
						data["datetime"] = pd.to_datetime(data[date_cols])

					else:
						data["datetime"] = pd.to_datetime(data[date_cols[0]] + " " + data[date_cols[1]])
				id_col = infer_id(data)
				data = data.set_index([id_col, "datetime"])
			if self._geom_cols is None:
				self._geom_cols = infer_geometry(data)

			data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(*[data[x] for x in self._geom_cols]),
			                        crs=self._crs)

		super().__init__(data)

		if columns_user_defined:
			data.columns = columns_user_defined

		if isinstance(data, gpd.GeoDataFrame):
			self.crs = self._crs
			self.geom_cols = self._geom_cols

	def to_geofeather(self, path):
		"""
		Writes TarjectoriesFrame to geofeather file
		:param path: write path
		:return: file path
		"""
		gf.to_geofeather(self.reset_index(), path, crs=self._crs)
		return path

	@classmethod
	def from_geofeather(self, path):
		"""
		Reads from geofeather file
		:param path: file path
		:return: TrajectoriesFrame object
		"""
		return TrajectoriesFrame(gf.from_geofeather(path))

	def to_shapefile(self, path):
		"""
		Writes TrajectoriesFrame to shapefile
		:param path: write path
		:return: file path
		"""
		warnings.warn("Saving to shapefile forces conversion of datetime into string")
		self['datetime'] = str(self['datetime'])
		gf.to_shp(self, path)
		return path

	def uloc(self, uid):
		"""
		Returns a part of TrajectoriesFrame for user ID
		:param uid: user id
		:return: TrajectoriesFrame
		"""
		return self.loc[uid]

	def get_users(self):
		"""
		Returns all the users' IDs
		:return: TrajectoriesFrame
		"""
		return self.index.get_level_values(0).unique().values
