top_path = """D:\\Projekty\\Sparse Chains\\markovian"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
from src.humobi.misc.generators import *
import os
from time import time
import json


df = TrajectoriesFrame(os.path.join(top_path,"markovian.csv")).loc[:20]
test_size = .2
split_ratio = 1 - test_size
train_frame, test_frame = split(df, split_ratio, 0)
test_lengths = test_frame.groupby(level=0).apply(lambda x: x.shape[0])
fname = open(os.path.join(top_path,'markovian_search_space_times.txt'),'w')
fname.write("20 users \n")

scores = {}
for n in [1,5,10,15,20,25,30,35,40,50,55,60,65,70,75,80,85,90,95,100]:
	lstart = time()
	sparse_alg = sparse_wrapper_learn(train_frame, overreach=True, reverse=True, old=False,
									  rolls=True, remove_subsets=False, reverse_overreach=True, jit=True,
									  search_size=n, parallel=True, cuda=False)
	lend = time()
	ltime = lend - lstart
	print(ltime)
	fname.write("Learn time %s %s \n" % (str(n),ltime))
	tstart = time()
	pred_res = sparse_wrapper_test(sparse_alg, test_frame, df, split_ratio, test_lengths,
							   length_weights = None, recency_weights=None,
							org_length_weights = None, org_recency_weights = None, use_probs=False)
	tend = time()
	ttime = tend - tstart
	fname.write("Pred time %s %s \n" % (str(n), ttime))
	scores[n] = pred_res

fname.close()

with open(os.path.join(top_path,'markovian_search_space.json'),'w') as ff:
	json.dump(scores,ff)