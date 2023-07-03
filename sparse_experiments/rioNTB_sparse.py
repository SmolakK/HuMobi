file_path = """D:\\processing_vanessa"""
save_path = """D:\\Projekty\\Sparse Chains\\RIO_NTB\\sparse_full"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
from src.humobi.misc.generators import *
import os
from time import time
import json

df = TrajectoriesFrame(os.path.join(file_path, "RIO_1H.csv"))
df = df[~df.labels.isna()]
df['labels'] = df.labels.astype(np.int64)
df = TrajectoriesFrame(df)
df = df.uloc(df.get_users())

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
                                  search_size=10, parallel=True)
cp = [('Q','IW','IWS','Q'),('Q','IW','IWS',None)]
for n in range(len(cp)):
    fname = open(os.path.join(save_path, 'rioNTB_sparse_times_%s.txt' % str(n+5)), 'w')
    lend = time()
    ltime = lend - lstart
    fname.write("Learn time %s \n" % (str(ltime)))
    tstart = time()
    forecast_df, pred_res, topk_res = sparse_wrapper_test(sparse_alg, test_frame, df.labels, split_ratio, test_lengths,
                                                length_weights=cp[n][0], recency_weights=cp[n][1],
                                                org_length_weights=cp[n][2], org_recency_weights=cp[n][3], use_probs=False)
    tend = time()
    ttime = tend - tstart
    fname.write("Pred time %s \n" % (str(ttime)))
    pred_res.to_csv(os.path.join(save_path, 'rioNTB_sparse_scores_%s.csv' % str(n+5)))
    forecast_df.to_csv(os.path.join(save_path, 'rioNTB_sparse_predictions_%s.csv' % str(n+5) ))
    with open(os.path.join(save_path,'rioNTB_sparse_topk_%s.csv' % str(n+5)),'w') as ff:
        json.dump(topk_res,ff)

fname.close()
