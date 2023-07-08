top_path = """D:\\Projekty\\Sparse Chains\\sparse_final"""
file_path = """D:\\Projekty\\Sparse Chains\\markovian"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
from src.humobi.misc.generators import *
import os
from time import time
import json


df = TrajectoriesFrame(os.path.join(file_path,"markovian.csv"))
fname = open(os.path.join(top_path,'markovian_sparse_final_times.txt'),'w')


test_size = .2
split_ratio = 1 - test_size
train_frame, test_frame = split(df, split_ratio, 0)
test_lengths = test_frame.groupby(level=0).apply(lambda x: x.shape[0])

lstart = time()
sparse_alg = sparse_wrapper_learn(train_frame, overreach=False, reverse=True, old=False,
								  rolls=True, remove_subsets=False, reverse_overreach=False,jit=True,
								  search_size=100, parallel=True, truncate=0.6)
lend = time()
ltime = lend - lstart
fname.write("Learn time %s \n" % (str(ltime)))

tstart = time()
forecast_df, pred_res, topk_res = sparse_wrapper_test(sparse_alg, test_frame, df.labels, split_ratio, test_lengths,
											length_weights=None, recency_weights=None,
											org_length_weights=None, org_recency_weights=None, use_probs=False,
													  uniqueness_weights='F',completeness_weights='Q')
tend = time()
ttime = tend - tstart
fname.write("Pred time %s \n" % (str(ttime)))
pred_res.to_csv(os.path.join(top_path, 'markovian_sparse_scores_100_U.csv'))
forecast_df.to_csv(os.path.join(top_path,'markovian_sparse_predictions_100_U.csv'))
with open(os.path.join(top_path,'markovian_sparse_topk_100_U'),'w') as ff:
    json.dump(topk_res,ff)

