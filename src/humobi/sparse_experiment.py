from src.humobi.misc.generators import markovian_sequences_generator, random_sequences_generator
from src.humobi.predictors.sparse import Sparse
import time
import numpy as np

models = {}
df = markovian_sequences_generator(1,3,600,1)
# df = random_sequences_generator(1,3,10)
for uid,vals in df.groupby(level=0):
    models[uid] = Sparse()
    models[uid].fit(df.labels.values)
    models[uid].predict(np.array([2,0,1,1,0,2,1,1]))