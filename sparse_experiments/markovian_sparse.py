top_path = """D:\\Projekty\\Sparse Chains\\Updated\\markovian\\sparse_val"""
file_path = """D:\\Projekty\\Sparse Chains\\markovian"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
from src.humobi.misc.generators import *
import os
from time import time
import json


df = TrajectoriesFrame(os.path.join(file_path,"markovian.csv"))
df = df.loc[df.get_users()[:20]]
fname = open(os.path.join(top_path,'markovian_sparse_times.txt'),'w')
fname.close()
comb_learn = [.3,.2]
# for q in range(2,9):
# 	comb_learn.append(q/10)
#

# comb_pred = []
# for x in [None,'L','Q']:
# 	for y in [None,'IW', 'IWS']:
# 		for z in [None,'IW', 'IWS']:
# 			for a in [None, 'L', 'Q']:
# 				for b in [None,'F', 'L', 'Q']:
# 					for c in [None, 'L', 'Q']:
# 						comb_pred.append((x, y, z, a, b, c))

comb_pred = []
for x in ['Q',None,'Q','L']:
	for y in ['Q',None,'IW','IWS']:
		for z in ['Q',None,'IW','IWS']:
			for a in ['Q',None,'Q','L']:
				for b in ['Q']:
					for c in ['Q']:
						comb_pred.append((x, y, z, a, b, c))

test_size = .2
split_ratio = 1 - test_size
valtrain_frame, test_frame = split(df, split_ratio, 0)
val_frame = valtrain_frame.groupby(level=0).apply(lambda x: x.iloc[round(len(x) * .8):]).droplevel(1)
train_frame = valtrain_frame.groupby(level=0).apply(lambda x: x.iloc[:round(len(x) * .8)]).droplevel(1)
test_lengths = val_frame.groupby(level=0).apply(lambda x: x.shape[0])

for cl in comb_learn:
	lstart = time()
	sparse_alg = sparse_wrapper_learn(train_frame, overreach=False, reverse=True, old=False,
									  rolls=True, remove_subsets=False, reverse_overreach=False,jit=True,
									  search_size=100, parallel=True, truncate = None)
	lend = time()
	ltime = lend - lstart
	with open(os.path.join(top_path, 'markovian_sparse_times.txt'), 'a') as ff:
		ff.write("Learn time %s \n" % (str(ltime)))
	for cp in comb_pred:
		tstart = time()
		forecast_df, pred_res, top_k = sparse_wrapper_test(sparse_alg, val_frame, valtrain_frame, split_ratio, test_lengths,
													length_weights=cp[0], recency_weights=cp[1],
													org_length_weights=cp[2], org_recency_weights=cp[3], use_probs=False,
													uniqueness_weights=cp[5],completeness_weights=cp[4])
		tend = time()
		ttime = tend - tstart
		with open(os.path.join(top_path,'markovian_sparse_times.txt'),'a') as ff:
			ff.write("Pred time %s %s \n" % (str(cp), str(ttime)))
		pred_res.to_csv(os.path.join(top_path, 'markovian_sparse_scores_%s_%s_%s.csv' % ('full',cp,cl)))
		forecast_df.to_csv(os.path.join(top_path,'markovian_sparse_predictions_%s_%s_%s.csv' % ('full',cp,cl)))
		with open(os.path.join(top_path, 'markovian_sparse_topk_%s_%s_%s.csv' % ('full',cp,cl)), 'w') as ff:
			json.dump(top_k, ff)


