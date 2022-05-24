import numpy as np
import pandas as pd
import geopandas as gpd
from src.humobi.models.spatial_tools.misc import normalize_array
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import math


def generate_points_from_distribution(distribution, amount):
	"""
	Randomly chooses points from given distributions
	:param distribution: The list with distribution
	:param amount: The number of points to choose
	:return: A vector of chosen points
	"""
	probabilities = normalize_array(distribution.counted.values.astype('float'))
	return np.random.choice(distribution.geometry.values, amount, p=probabilities)


def select_points_with_commuting(starting_positions, target_distribution, commuting, spread=None):
	"""
	Returns an array with work positions
	:param starting_positions: contains home positions
	:param target_distribution: contains workplace distribution
	:param commuting: contains commuting distance distribution by each cell
	:param spread: an optional parameter, the value of spread to create an annulus
	:return: an array with work positions
	"""
	target_distribution = target_distribution[['geometry', 'counted']]
	distance_indicies = [indeks[0] for indeks in [
		commuting[1][commuting[1]['geometry'] == home].index.values for home in starting_positions]]
	distances = commuting[1].loc[distance_indicies]
	if spread is None:
		buffers = [starting_positions[dist].buffer(distances.values[dist][0]) for dist in range(len(distances))]
	else:
		buffers = [starting_positions[dist].buffer(distances.values[dist][0] * (1.0 + spread)).difference(
			starting_positions[dist].buffer(distances.values[dist][0] * (1.0 - spread))) for dist in
			range(len(distances))]
	chosen_work_places = []
	index = 0
	for buffer in buffers:
		if buffer.area == 0.0:
			chosen = None
		# chosen = gpd.GeoDataFrame(pd.DataFrame(starting_positions), geometry = 0).loc[index][0]  # decide on the option
		else:
			inside_buffer = target_distribution.loc[target_distribution['geometry'].intersects(buffer) == True]
			if len(inside_buffer) != 0 and not np.all(inside_buffer.values[:, 1] == 0):
				probabilities = normalize_array(inside_buffer.values[:, 1].astype('float'))
				chosen = np.random.choice(inside_buffer.values[:, 0], size=1, p=probabilities)[0]
			else:
				chosen = None
		chosen_work_places.append(chosen)
		index += 1
	return np.array(chosen_work_places)


def create_ellipse(home_location, work_location, spread):
	"""
	Returns ellipse with a home at its centre and work location at the edge
	:param home_location: contains a home position;
	:param work_location: contains a work position;
	:param spread: contains a ratio between major and minor axis;
	:return: ellipse with a home at its centre and work location at the edge.
	"""
	if work_location is None:
		return None
	else:
		a = home_location.distance(work_location)
		b = a * float(spread)
		point_list = []
		azimuth = math.atan2(work_location.y - home_location.y, work_location.x - home_location.x)
		ro = (math.pi / 200)

		for t in range(0, 401):
			x = home_location.x + (a * math.cos(t * ro) * math.cos(azimuth) - b * math.sin(t * ro) * math.sin(azimuth))
			y = home_location.y + (b * math.sin(t * ro) * math.cos(azimuth) + a * math.cos(t * ro) * math.sin(azimuth))
			point_list.append([Point(x, y).x, Point(x, y).y])
		return Polygon(point_list)


def generate_activity_areas(area_type, home_positions, work_positions, layer, spread):
	"""
	Returns an array with acitvity areas
	:param area_type: the type of activity area
	:param home_positions: contains home locations
	:param work_positions: contains work locations
	:param spread: contains a ratio between major and minor axis
	:return: an array with acitvity areas
	"""
	activity_areas = []
	if area_type == 'ellipse':
		for index in range(0, len(home_positions)):
			ellipse = create_ellipse(home_positions[index], work_positions[index], spread)
			try:
				intersection_layer = dict(layer.intersects(ellipse))
				gridded_activity_area = {k: v for k, v in intersection_layer.items() if v == True}
				activity_area = [layer.loc[n] for n in list(gridded_activity_area.keys())]
				activity_areas.append(activity_area)
			except:
				activity_areas.append(None)
		return np.array(activity_areas)  # return activity_areas
	else:
		pass
