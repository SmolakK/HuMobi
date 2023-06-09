top_path = """D:\\Projekty\\Sparse Chains\\sparse"""
file_path = """D:\\Projekty\\Sparse Chains\\markovian"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
from src.humobi.misc.generators import *
import os
from time import time
import json


df = TrajectoriesFrame(os.path.join(file_path,"markovian.csv"))
fname = open(os.path.join(top_path,'markovian_sparse_times.txt'),'w')
comb_learn = []
for x in [True,False]:
	for y in [True,False]:
		for z in [True,False]:
			for a in [False]:
				for b in [True,False]:
					for n in range(20,100,10):
						comb_learn.append((x,y,z,a,b,n))
comb_pred = []
for x in [None,'L','Q','IW','IWS']:
	for y in [None, 'L', 'Q', 'IW', 'IWS']:
		for z in [None, 'L', 'Q', 'IW', 'IWS']:
			for a in [None, 'L', 'Q', 'IW', 'IWS']:
				for b in [False,True]:
					comb_pred.append((x,y,z,a,b))

test_size = .2
split_ratio = 1 - test_size
train_frame, test_frame = split(df, split_ratio, 0)
test_lengths = test_frame.groupby(level=0).apply(lambda x: x.shape[0])
for c in comb_learn:
	lstart = time()
	sparse_alg = sparse_wrapper_learn(train_frame, overreach=c[0], reverse=c[1], old=False,
	                                  rolls=c[2], remove_subsets=c[3], reverse_overreach=c[4],jit=True,
	                                  search_size=c[5], parallel=True)
	lend = time()
	ltime = lend - lstart
	fname.write("Learn time %s %s \n" % (str(c), str(ltime)))
	for cp in comb_pred:
		tstart = time()
		forecast_df, pred_res = sparse_wrapper_test(sparse_alg, test_frame, df, split_ratio, test_lengths,
		                               length_weights=cp[0], recency_weights=cp[1],
		                               org_length_weights=cp[2], org_recency_weights=cp[3], use_probs=cp[4])
		tend = time()
		ttime = tend - tstart
		fname.write("Pred time %s %s \n" % (str(cp), str(ttime)))
		pred_res.to_csv(os.path.join(top_path, 'markovian_sparse_scores_%s_%s.csv' % (c,cp)))
		forecast_df.to_csv(os.path.join(top_path,'markovian_sparse_predictions_%s_%s.csv' % (c,cp)))

fname.close()
