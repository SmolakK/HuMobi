from src.humobi.misc.generators import markovian_sequences_generator, random_sequences_generator
from src.humobi.predictors.sparse import Sparse, Sparse_old
from src.humobi.misc.utils import _equally_sparse_match
import time
import numpy as np
models = {}
df = markovian_sequences_generator(1,3,200,1)
# df = random_sequences_generator(1,3,10)
for uid,vals in df.groupby(level=0):
    models[uid] = Sparse_old()
    models[uid].fit(np.array([2,0,1,1,0,2,1]))
    models[uid] = Sparse(reverse=True,overreach=True,rolls=False)
    models[uid].fit(np.array([2, 0, 1, 1, 0, 2, 1]))
    models[uid].predict(np.array([2,0,1,1,0,2,1]))