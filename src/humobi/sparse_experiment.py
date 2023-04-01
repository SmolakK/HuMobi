from src.humobi.misc.generators import markovian_sequences_generator, random_sequences_generator
from src.humobi.predictors.sparse import Sparse
import time
import numpy as np

models = {}
# df = markovian_sequences_generator(1,3,6,1)
df = random_sequences_generator(1,3,400)
for uid,vals in df.groupby(level=0):
    models[uid] = Sparse()
    start = time.time()
    models[uid].fit(vals.labels.values)
    end = time.time()
    print(end-start)
    start = time.time()
    models[uid].fit2(vals.labels.values)
    end = time.time()
    print(end - start)
    start = time.time()
    models[uid].fit3(vals.labels.values)
    end = time.time()
    print(end - start)
    models[uid]