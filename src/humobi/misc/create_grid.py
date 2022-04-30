from src.humobi.misc.utils import resolution_to_points, moving_average
import numpy as np
from math import ceil
from shapely import geometry
import geopandas as gpd


def create_grid(trajectories_frame, resolution):
	"""
	Creates grid fitting given trajectories in TrajectoriesFrame class object.

	Args:
		trajectories_frame: TrajectoriesFrame class object
		resolution: grid resolution

	Returns:
		A GeoDataFrame with the grid
	"""
	crs= trajectories_frame.crs
	_x_min = trajectories_frame['geometry'].x.min()
	_x_max = trajectories_frame['geometry'].x.max()
	_y_min = trajectories_frame['geometry'].y.min()
	_y_max = trajectories_frame['geometry'].y.max()
	x_points = resolution_to_points(resolution, _x_max, _x_min)
	y_points = resolution_to_points(resolution, _y_max, _y_min)
	x = np.linspace(_x_min, _x_max, ceil(x_points))
	y = np.linspace(_y_min, _y_max, ceil(y_points))
	grid_cells = []
	for xn in range(len(x)-1):
		for yn in range(len(y)-1):
			grid_cells.append(geometry.box(x[xn], y[yn], x[xn+1], y[yn+1]))
	return gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
