from src.humobi.misc.generators import markovian_sequences_generator
from src.humobi.predictors.sparse import Sparse

models = {}
df = markovian_sequences_generator(10,10,1000,[x/10 for x in range(1,10)])
for uid,vals in df.groupby(level=0):
    models[uid] = Sparse(vals.labels.values)
    models[uid].predict([vals.labels.values])