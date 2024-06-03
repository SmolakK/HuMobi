import geopandas as gpd


def filter_layer(layer, trajectories):
    """
    Filters the aggregation layer leaving only cells when an observations are present.
    :param layer: Aggregation layer (could be a GeoDataFrame or an object with a 'layer' attribute that is a GeoDataFrame)
    :param trajectories: TrajectoriesFrame class object
    :return: A filtered aggregation layer
    """
    if hasattr(layer, 'layer'):
        layer_gdf = layer.layer
    elif isinstance(layer, gpd.GeoDataFrame):
        layer_gdf = layer
    else:
        raise TypeError("The 'layer' parameter must be a GeoDataFrame or an object with a 'layer' attribute that is a GeoDataFrame.")

    # Perform spatial join and filter the layer
    filtered_layer = gpd.sjoin(layer_gdf, trajectories, how='inner')
    return filtered_layer[layer_gdf.columns].drop_duplicates()