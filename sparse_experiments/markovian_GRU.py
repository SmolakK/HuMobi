from src.humobi.predictors.deep import *
top_path = """D:\\Projekty\\Sparse Chains\\sparse"""
file_path = """D:\\Projekty\\Sparse Chains\\markovian"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
from src.humobi.misc.generators import *
import os
from time import time
import json


df = TrajectoriesFrame(os.path.join(file_path,"markovian.csv"))

# DEEP LEARNING METHODS
GRU = DeepPred("GRU", df, test_size=.2, folds=1, window_size=5, batch_size=5, embedding_dim=8,
                rnn_units=1024)
GRU.learn_predict()
GRU_score = GRU.scores
print(GRU_score)