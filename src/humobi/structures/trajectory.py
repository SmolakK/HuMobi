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
	elif isinstance(date_col, str) or isinstance(date_col, int) and time_col is None:
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
		except:
			pass
		try:
			longitude_column = [x for x in trajectories_frame.columns if 'lon' in x.lower()][0]
		except:
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
	return longitude_column, latitude_column


def infer_id(trajectories_frame):
	"""
	This function infers the index of column with user identifier

	Args:
		trajectories_frame (TrajectoriesFrame): TrajectoriesFrame class object

	Returns:
		identifier column index
	"""
	if trajectories_frame.columns.dtype is 'str' or trajectories_frame.columns.is_object():
		try:
			id_col = [x for x in trajectories_frame.columns if 'id' in x.lower()][0]
			return id_col
		except:
			samples = trajectories_frame.sample(1000,replace=True)
			id_col = np.argmax([samples.groupby(x).size().mean() for x in samples.columns])
			id_col = trajectories_frame.columns[id_col]
			warnings.warn("ID column was infered but to avoid mistakes ensure that TrajectoriesFrame contains an ID "
			              "column (with id in the name) and reread the frame!")
			return id_col


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

		Args:
			data: DataFrame or file path
			params: dictionary of parameters accepted by DataFrame class
		"""
		try:
			self._crs = data.crs
		except AttributeError:
			passed_crs = params.pop('crs', None)
			self._crs = passed_crs

		if isinstance(data, str):
			data = pd.read_csv(data, **params)

		columns_user_defined = params.pop('columns', None)
		self._geom_cols = params.pop('geom_cols', None)

		if isinstance(data, pd.DataFrame):
			if not data.index.nlevels > 1:
				if len(data.select_dtypes(include=[np.datetime64])) > 1:
					if 'datetime' in data.columns:
						date_cols = 'datetime'
					elif 'time' in data.columns:
						date_cols = 'time'
					else:
						date_cols = infer_datetime(data)
					if isinstance(date_cols, str) or isinstance(date_cols, int):
						data["datetime"] = pd.to_datetime(data[date_cols])
					else:
						data["datetime"] = pd.to_datetime(data[date_cols[0]] + " " + data[date_cols[1]])
					if date_cols != 'datetime':
						data = data.drop(date_cols, axis=1)
				id_col = infer_id(data)
				if id_col != 'user_id':
					data['user_id'] = data[id_col]
					data = data.drop(id_col,axis=1)
				data = data.set_index(['user_id', "datetime"])
			if self._geom_cols is None:
				self._geom_cols = infer_geometry(data)

			data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(*[data[x] for x in self._geom_cols[::-1]]),
			                        crs=self._crs)

		super().__init__(data)

		if columns_user_defined:
			data.columns = columns_user_defined

		if isinstance(data, gpd.GeoDataFrame):
			if self._crs is None:
				self.crs = passed_crs
			self.geom_cols = self._geom_cols

	@property
	def _base_class_view(self):
		"""
		Forces DataFrame view in debugging mode for some IDEs.
		"""
		return pd.DataFrame(self)

	def to_geofeather(self, path):
		"""
		Writes TarjectoriesFrame to geofeather file

		Args:
			path: write path

		Returns:
			file path
		"""
		gf.to_geofeather(self.reset_index(), path)
		return path

	@classmethod
	def from_geofeather(self, path):
		"""
		Reads from geofeather file

		Args:
			path: file path

		Returns:
			TrajectoriesFrame object
		"""
		return TrajectoriesFrame(gf.from_geofeather(path))

	def to_shapefile(self, path):
		"""
		Writes TrajectoriesFrame to shapefile

		Args:
			path: write path

		Returns:
			file path
		"""
		warnings.warn("Saving to shapefile forces conversion of datetime into string")
		self = self.reset_index()
		self['datetime'] = str(self['datetime'])
		self.to_file(path)
		return path

	def uloc(self, uid):
		"""
		Returns a part of TrajectoriesFrame for user ID

		Args:
			uid: user id

		Returns:
			TrajectoriesFrame of selected users
		"""
		return self.loc[uid]

	def get_users(self):
		"""
		Returns all the users' IDs

		Returns:
			Array of all users' identifiers
		"""
		return self.index.get_level_values(0).unique().values

	def to_crs(self, dest_crs, cur_crs=None):
		"""
		Transforms geometry of TrajectoryFrame into different Coordinate Reference System.

		Args:
			dest_crs: CRS to which TrajectoriesFrame will be converted
			cur_crs (default=None): Current CRS of a TrajectoryFrame. If not determined, CRS will be read from the metadata.

		Returns:
			Converted TrajectoryFrame. This also alters the instance itself.
		"""
		if cur_crs is None:
			cur_crs = self.crs
		self = super(TrajectoriesFrame, self).to_crs(dest_crs)
		self[[x for x in self._geom_cols]] = pd.concat([self.geometry.x, self.geometry.y], axis=1)
		return self
