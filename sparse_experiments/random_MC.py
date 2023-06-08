top_path = """D:\\Projekty\\Sparse Chains\\markovian"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
from src.humobi.misc.generators import *
import os
from time import time
import json


df = TrajectoriesFrame(os.path.join(top_path,"random.csv"))
test_size = .2
split_ratio = 1 - test_size
train_frame, test_frame = split(df, split_ratio, 0)
test_lengths = test_frame.groupby(level=0).apply(lambda x: x.shape[0])
fname = open(os.path.join(top_path,'random_MC_times.txt'),'w')

# MARKOV CHAINS
for state_size in range(10):
	start = time()
	MC1 = markov_wrapper(df, test_size=.2, state_size=state_size, update=False, online=True)
	end = time()
	print(end-start)
	fname.write("MC%s: %s\n" % (str(state_size),str(end-start)))
	MC1.to_csv(open(os.path.join(top_path,'random_MC'+str(state_size)+'.csv'),'w'))
	print(MC1.mean())

fname.close()