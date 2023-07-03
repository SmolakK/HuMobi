file_path = """D:\\processing_vanessa"""
save_path = """D:\\Projekty\\Sparse Chains\\RIO_NTB"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
from src.humobi.misc.generators import *
import os
from time import time
import json


df = TrajectoriesFrame(os.path.join(file_path,"RIO_1H.csv"))
df = df[~df.labels.isna()]
test_size = .2
split_ratio = 1 - test_size
train_frame, test_frame = split(df, split_ratio, 0)
test_lengths = test_frame.groupby(level=0).apply(lambda x: x.shape[0])
fname = open(os.path.join(save_path,'MC_times.txt'),'w')

# MARKOV CHAINS
for state_size in range(10):
	start = time()
	predictions, MC1, topk = markov_wrapper(df, test_size=.2, state_size=state_size, update=False, online=True)
	end = time()
	print(end-start)
	fname.write("MC%s: %s\n" % (str(state_size),str(end-start)))
	MC1.to_csv(open(os.path.join(save_path,'MC'+str(state_size)+'.csv'),'w'))
	predictions.to_csv(open(os.path.join(save_path,'predictons_MC'+str(state_size)+'.csv'),'w'))
	print(MC1.mean())
	with open(os.path.join(save_path,'topk_MC'+str(state_size)+'.csv'),'w') as ff:
		json.dump(topk,ff)

fname.close()