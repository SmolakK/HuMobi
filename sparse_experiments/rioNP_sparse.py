file_path = """D:\\processing_vanessa"""
save_path = """D:\\Projekty\\Sparse Chains\\RIO_NP\\sparse_full"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
from src.humobi.misc.generators import *
import os
from time import time
import json

df = TrajectoriesFrame(os.path.join(file_path, "RIO_NP.csv"))
df['labels'] = df.labels.astype(np.int64)
df = df.uloc(df.get_users())
fname = open(os.path.join(save_path, 'rioNP_sparse_times_100.txt'), 'w')

test_size = .2
split_ratio = 1 - test_size
valtrain_frame, test_frame = split(df, split_ratio, 0)
train_frame = valtrain_frame
# val_frame = valtrain_frame.groupby(level=0).apply(lambda x: x.iloc[round(len(x) * .8):]).droplevel(1)
# train_frame = valtrain_frame.groupby(level=0).apply(lambda x: x.iloc[:round(len(x) * .8)]).droplevel(1)
test_lengths = test_frame.groupby(level=0).apply(lambda x: x.shape[0])
lstart = time()
sparse_alg = sparse_wrapper_learn(train_frame, overreach=False, reverse=True, old=False,
                                  rolls=True, remove_subsets=False, reverse_overreach=False, jit=True,
                                  search_size=100, parallel=True)
lend = time()
ltime = lend - lstart
fname.write("Learn time %s \n" % (str(ltime)))
tstart = time()
forecast_df, pred_res, topk_res = sparse_wrapper_test(sparse_alg, test_frame, df.labels, split_ratio, test_lengths,
                                            length_weights=None, recency_weights='IW',
                                            org_length_weights='IW', org_recency_weights=None, use_probs=False)
tend = time()
ttime = tend - tstart
fname.write("Pred time %s \n" % (str(ttime)))
pred_res.to_csv(os.path.join(save_path, 'rioNP_sparse_scores_100'))
forecast_df.to_csv(os.path.join(save_path, 'rioNP_sparse_predictions_100'))
with open(os.path.join(save_path,'rioNP_sparse_topk_100'),'w') as ff:
    json.dump(topk_res,ff)

fname.close()
