import sys
sys.path.append('.')
import numpy as np
from shapely.geometry import Point
from src.humobi.misc.utils import resolution_to_points, moving_average, to_labels
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import concurrent.futures as cf
from itertools import repeat
from src.humobi.structures.trajectory import TrajectoriesFrame


class GridAggregation():
	"""
	A class for spatial aggregation of movement trajectories. This one uses a regular grid.
	"""

	def __init__(self, resolution=300, origin=True, x_min=None, x_max=None, y_min=None, y_max=None):
		"""
		Class initialisation. Depending on what is defined, the grid fits to the given data spatial extent of in the middle
		of data.

		Args:
			resolution (default = 300): grid resolution
			origin: Determines if the grid should be centered to the centroid of the data
			x_min: minimum x value of the grid
			x_max: maximum x value of the grid
			y_min: minimum y value of the grid
			y_max: maximum y value of the grid
		"""
		self._x_min = x_min
		self._x_max = x_max
		self._y_min = y_min
		self._y_max = y_max

		self._resolution = resolution
		if isinstance(origin, bool):
			self._detect_origin = True
		else:
			self._detect_origin = False
		if x_min is None or x_max is None or y_min is None or y_max is None:
			self._detect_range = True
		else:
			self._detect_range = False
		if not self._detect_origin and not self._detect_range:
			raise AttributeError("Cannot set origin and range at the same time!")

	@property
	def resolution(self):
		return self._resolution

	@property
	def x_min(self):
		return self._x_min

	@property
	def x_max(self):
		return self._x_max

	@property
	def y_min(self):
		return self._y_min

	@property
	def y_max(self):
		return self._y_max

	def _aggregate_multi(self, x, y, centroid_x, centroid_y, row_index, val):
		"""
		Fast spatial aggregation of movement trajectories spatial aggregation. Multithread variant.

		Args:
			x: grid x coordinates
			y: grid y coordinates
			centroid_x: grid x coordinates of centroids
			centroid_y: grid y coordinates of centroids
			row_index: data row index
			val: data row value

		Returns:
			assigned geometry and row index
		"""
		lower_x = np.max(np.where((val.x >= x)))
		lower_y = np.max(np.where((val.y >= y)))
		if lower_x == len(centroid_x):
			assigned_x = centroid_x[lower_x - 1]
		else:
			assigned_x = centroid_x[lower_x]
		if lower_y == len(centroid_y):
			assigned_y = centroid_y[lower_y - 1]
		else:
			assigned_y = centroid_y[lower_y]
		return row_index, Point(assigned_x, assigned_y)

	def _aggregate(self, x, y, centroid_x, centroid_y, val):
		"""
		Fast movement trajectories spatial aggregation
		Args:
			x: grid x coordinates
			y: grid y coordinates
			centroid_x: grid x coordinates of centroids
			centroid_y: grid y coordinates of centroids
			row_index: data row index
			val: data row value

		Returns:
			assigned geometry and row index
		"""
		lower_x = np.max(np.where((val.x >= x)))
		lower_y = np.max(np.where((val.y >= y)))
		if lower_x == len(centroid_x):
			assigned_x = centroid_x[lower_x - 1]
		else:
			assigned_x = centroid_x[lower_x]
		if lower_y == len(centroid_y):
			assigned_y = centroid_y[lower_y - 1]
		else:
			assigned_y = centroid_y[lower_y]
		return assigned_x, assigned_y

	def aggregate(self, trajectories_frame, inplace=True, parallel=False):
		"""
		Aggregates TrajectoriesFrame object spatially.

		Args:
			trajectories_frame: TrajectoriesFrame class object
			inplace: determines if coordinates should be overwritten. If false, a new column is added.
			parallel: determines if should uses multithreading (default = False, because it is slower for small data)

		Returns:
			A TrajectoriesFrame with aggregated geometry.
		"""
		trajectories_copy = trajectories_frame.copy()
		if self._detect_range and self._detect_origin:
			self._x_min = trajectories_frame['geometry'].x.min()
			self._x_max = trajectories_frame['geometry'].x.max()
			self._y_min = trajectories_frame['geometry'].y.min()
			self._y_max = trajectories_frame['geometry'].y.max()
		x_points = resolution_to_points(self.resolution, self.x_max, self.x_min)
		y_points = resolution_to_points(self.resolution, self.y_max, self.y_min)
		x = np.linspace(self.x_min, self.x_max, x_points)
		y = np.linspace(self.y_min, self.y_max, y_points)
		centroid_x = moving_average(x)
		centroid_y = moving_average(y)
		agg_dict = {}
		if parallel:
			with cf.ThreadPoolExecutor(max_workers=6) as executor:
				indis = [indi for indi, val in trajectories_frame['geometry'].iteritems()]
				vals = [val for indi, val in trajectories_frame['geometry'].iteritems()]
				results = list(tqdm(
					executor.map(self._aggregate_multi, repeat(x), repeat(y), repeat(centroid_x), repeat(centroid_y),
					             indis, vals), total=trajectories_frame.shape[0]))
			for result in results:
				agg_dict[result[0]] = result[1]
		else:
			for indi, val in tqdm(trajectories_frame['geometry'].iteritems(), total=trajectories_frame.shape[0]):
				if np.isnan(val.x):
					agg_dict[indi] = Point(float('nan'), float('nan'))
					continue
				assigned_x, assigned_y = self._aggregate(x, y, centroid_x, centroid_y, val)
				agg_dict[indi] = Point(assigned_x, assigned_y)
		if inplace:
			trajectories_copy['geometry'].update(pd.Series(agg_dict))
			trajectories_copy[trajectories_copy._geom_cols[0]] = trajectories_copy["geometry"].x
			trajectories_copy[trajectories_copy._geom_cols[1]] = trajectories_copy["geometry"].y
		else:
			trajectories_copy['old_geometry'] = trajectories_copy['geometry']
			trajectories_copy['geometry'] = pd.Series(agg_dict)
			trajectories_copy[trajectories_copy._geom_cols[0]] = trajectories_copy["geometry"].x
			trajectories_copy[trajectories_copy._geom_cols[1]] = trajectories_copy["geometry"].y
			trajectories_copy['geometry'] = trajectories_copy['old_geometry']
		trajectories_copy = to_labels(trajectories_copy)
		return trajectories_copy


class ClusteringAggregator():
	"""
	A class for spatial aggregation of movement trajectories. This one uses clustering algorithms from sklearn library.
	"""

	def __init__(self, algorithm, **kwargs):
		"""
		Class initialisation. Accepts sklearn clustering algorithms' classes and their keyword arguments.

		Args:
			algorithm: Clustering algorithm from sklearn library.
			**kwargs: Any accepted kwargs.
		"""
		self._algorithm = algorithm(**kwargs)

	@property
	def algorithm(self):
		return self._algorithm

	def _user_aggregate(self, single_trajectory):
		"""
		Spatially aggregates single movement trajectory. Adds labels columns.

		Args:
			single_trajectory: Single movement trajectory

		Returns:
			An aggregated TrajectoriesFrame with labels column added.
		"""
		try:
			fited = self.algorithm.fit(single_trajectory)
		except:
			raise ValueError
		single_trajectory['labels'] = fited.labels_
		return single_trajectory

	def _recalcuate_centres(self, single_trajectory):
		"""
		Based on labels, recalculates spatial coordinates of points to their clusters centers.

		Args:
			single_trajectory: single movement trajectory

		Returns:
			a TrajectoriesFrame with overwritten coordinates in lon and lat columns
		"""
		centres = single_trajectory.groupby(by='labels').apply(lambda x: x.mean())
		single_trajectory = single_trajectory.join(centres, on='labels', rsuffix='_')
		single_trajectory['lat'] = single_trajectory['lat_']
		single_trajectory['lon'] = single_trajectory['lon_']
		return single_trajectory

	def aggregate(self, trajectories_frame, drop_noise=False, centres_as_geometry=True):
		"""
		General function for spatial data aggregation. It assigns labels to clusters and has additional options of
		centers recalculation and data overwriting.

		Args:
			trajectories_frame: TrajectoriesFrame object class
			drop_noise: Datermines if noise (-1 labels) should be drop
			centres_as_geometry: If true, centres of clusters are assigned as geometry
		Raturns:
			A spatially aggregated TrajectoriesFrame
		"""
		if not hasattr(trajectories_frame, '_geom_cols'):
			trajectories_frame = TrajectoriesFrame(trajectories_frame)
		coordinates_frame = trajectories_frame[[trajectories_frame._geom_cols[0], trajectories_frame._geom_cols[1]]]
		clustered = coordinates_frame.groupby(level=0).progress_apply(lambda x: self._user_aggregate(x))
		if drop_noise:
			clustered[clustered['labels'] == -1] = None
			if len(clustered) == 0:
				return None
		if centres_as_geometry:
			clustered = clustered.groupby(level=0).progress_apply(lambda x: self._recalcuate_centres(x))
			clustered['geometry'] = [Point(*x) for x in
			                         zip(clustered.loc[:, 'lat'],
			                             clustered.loc[:, 'lon'])]
		merged = pd.merge(clustered, trajectories_frame, left_index=True, right_index=True, how='outer',
		                  suffixes=['', 'y'])
		merged = merged[['lat', 'lon', 'geometry', 'labels', 'start', 'end']]
		merged[merged['labels'] == -1] = None
		trajectories_frame = TrajectoriesFrame(merged)
		return trajectories_frame


class LayerAggregator():
	"""
	A class for movement trajectories spatial aggregation. This one uses external data (spatial layer) to aggregate data.
	"""

	def __init__(self, layer, kwargs={}):
		"""
		Class initalisation.

		Args:
		layer: An external, spatial layer which will be used for aggregation
		"""
		self._layer = gpd.GeoDataFrame.from_file(layer, **kwargs)

	@property
	def layer(self):
		return self._layer

	@layer.setter
	def layer(self, new):
		self._layer = new

	def aggregate(self, trajectories_frame, dropoverwirte=True):
		"""
		Aggregates data to given spatial layer. Overwrites geometry.

		Args:
			trajectories_frame: TrajectoriesFrame object class
			dropoverwirte: If true, points that were not assigned to any locationare removed.

		Returns:
			An aggregated TrajectoriesFrame instance
		"""
		self._layer = self.layer.to_crs(trajectories_frame.crs)
		aggregated_frame = gpd.tools.sjoin(trajectories_frame.reset_index(), self.layer, how='left')
		centroid_layer = self.layer['geometry'].centroid
		aggregated_frame = aggregated_frame[["user_id", "datetime", 'index_right']]
		aggregated_frame = pd.merge(aggregated_frame, pd.DataFrame(centroid_layer), left_on='index_right',
		                            right_index=True)
		if dropoverwirte:
			trajectories_frame = trajectories_frame.join(aggregated_frame.set_index(["user_id", "datetime"]), how='inner')
		else:
			trajectories_frame = trajectories_frame.join(aggregated_frame.set_index(["user_id", "datetime"]))
		trajectories_frame['geometry'] = trajectories_frame.iloc[:, -1]
		trajectories_frame = trajectories_frame[trajectories_frame.columns[:-1]]
		trajectories_frame['lon'] = trajectories_frame.geometry.x
		trajectories_frame['lat'] = trajectories_frame.geometry.y
		trajectories_frame = TrajectoriesFrame(trajectories_frame)
		return trajectories_frame
