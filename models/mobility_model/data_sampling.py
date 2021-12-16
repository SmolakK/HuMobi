import pandas as pd
import geopandas as gpd
import sys
from misc import create_layer
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
sys.path.append("..")
from misc.create_layer import create_grid
from misc.utils import normalize