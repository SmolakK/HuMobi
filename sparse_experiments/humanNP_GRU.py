from src.humobi.predictors.deep import *
top_path = """D:\\Projekty\\Sparse Chains\\sparse_NP"""
file_path = """D:\\Projekty\\Sparse Chains\\markovian"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
from src.humobi.misc.generators import *
import os
from time import time
import json

df = TrajectoriesFrame("D:\\Projekty\\bias\\london\\london_seq_111.7572900082951_1.csv",
                       {'names':['id','datetime','lat','lon','geometry','labels','start','end'],"skiprows":1})
df['labels'] = df.labels.astype(np.int64)
df = df.uloc(df.get_users()[:100])


# DEEP LEARNING METHODS
GRU = DeepPred("GRU", df, test_size=.2, folds=1, window_size=5, batch_size=10, embedding_dim=8,
                rnn_units=1024)
GRU.learn_predict()
GRU_score = GRU.scores
print(GRU_score)