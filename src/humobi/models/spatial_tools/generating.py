import numpy as np
import pandas as pd
import geopandas as gpd
from src.humobi.models.spatial_tools.misc import normalize_array
from shapely.geometry import Point, Polygon
import math


def generate_points_from_distribution(distribution, amount):
    """
    Randomly chooses points from given distributions
    :param distribution: The list with distribution
    :param amount: The number of points to choose
    :return: A vector of chosen points
    """
    choices = []
    probabilities = distribution.counted.values.astype('float')
    for n in range(amount):
        choose_from = normalize_array(probabilities)
        choice = np.random.choice(distribution.index.values, p=choose_from)
        choices.append(distribution.loc[choice].geometry)
        wherewasit = np.argwhere(distribution.index.values == choice)[0][0]
        probabilities[wherewasit] -= 1 / amount
        probabilities[probabilities < 0] = 0
    return gpd.GeoSeries(choices)


def select_points_with_commuting(home_locations, target_distribution, commuting, spread=None):
    """
    Returns an array with work positions
    :param starting_positions: contains home positions
    :param target_distribution: contains workplace distribution
    :param commuting: contains commuting distance distribution by each cell
    :param spread: an optional parameter, the value of spread to create an annulus
    :return: an array with work positions
    """
    starting_positions = []
    starting_positions.append(home_locations)
    for n_place in range(1,len(commuting)+1):
        current_target_distribution = target_distribution[n_place][['geometry', 'counted']].copy()
        current_commuting_distribution = commuting[n_place].copy()
        buffer_nplaces = []
        for start_place in range(n_place):
            commuting_distribution_from_place = gpd.GeoDataFrame(current_commuting_distribution.iloc[:,start_place*2],
                                                                 geometry=current_commuting_distribution.iloc[:,(start_place+1)*2-1])
            commuting_start = gpd.GeoDataFrame(starting_positions[start_place],geometry=starting_positions[start_place])
            distance_from_point = commuting_distribution_from_place.sjoin(commuting_start,how='right')
            if spread is None:
                buffers = distance_from_point.buffer(distance_from_point['distance']*1.01)
            else:
                buffers = distance_from_point.buffer(distance_from_point['distance']*(1.0+spread)).difference(
                    distance_from_point.buffer(distance_from_point['distance']*(1.0-spread)))
            buffer_nplaces.append(buffers)
        main_buffer = buffer_nplaces[0]
        for buffer in buffer_nplaces:
            main_buffer = main_buffer.intersection(buffer)
        buffers = main_buffer
        chosen_work_places = []
        ind = 0
        for buffer in buffers:
            if buffer.area == 0.0:
                # chosen = None
                chosen = gpd.GeoDataFrame(pd.DataFrame(home_locations), geometry=0).loc[ind][
                    0]  # decide on the option
            else:
                inside_buffer = current_target_distribution.loc[current_target_distribution['geometry'].intersects(buffer) == True]
                if len(inside_buffer) != 0 and not np.all(inside_buffer.values[:, 1] == 0):
                    probabilities = normalize_array(inside_buffer.values[:, 1].astype('float'))
                    chosen = np.random.choice(inside_buffer.values[:, 0], size=1, p=probabilities)[0]
                    where = current_target_distribution[current_target_distribution.geometry == chosen].index
                    current_target_distribution.loc[where, 'counted'] = current_target_distribution.loc[where, 'counted'] - 1 / len(buffers)
                    current_target_distribution.loc[current_target_distribution.counted < 0, 'counted'] = 0
                else:
                    chosen = gpd.GeoDataFrame(pd.DataFrame(home_locations), geometry=0).loc[ind][
                        0]  # decide on the option
            chosen_work_places.append(chosen)
            ind += 1
        starting_positions.append(gpd.GeoSeries(chosen_work_places))
    sig_places_chosen = starting_positions[1:]
    return sig_places_chosen


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
        for inde in range(0, len(home_positions)):
            ellipse = create_ellipse(home_positions[inde], work_positions[0][inde], spread)
            activity_areas.append(ellipse)
    elif 'convex' in area_type:
        all_locations = pd.concat([home_positions,pd.concat(work_positions,axis=1)],axis=1)
        all_locations = all_locations.T.reset_index(drop=True).T
        all_locations_main = all_locations.iloc[:,0]
        for col in all_locations.columns[1:]:
            all_locations_main = all_locations_main.union(all_locations.loc[:,col])
        activity_areas = all_locations_main.convex_hull
    else:
        pass
    return activity_areas  # return activity_areas