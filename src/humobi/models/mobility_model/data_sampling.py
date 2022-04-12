import pandas as pd
import geopandas as gpd
import sys
from src.humobi.misc import create_layer
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
sys.path.append("..")
from src.humobi.misc.create_layer import create_grid
from src.humobi.misc.utils import normalize