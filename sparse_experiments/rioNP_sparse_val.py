file_path = """D:\\processing_vanessa"""
save_path = """D:\\\Projekty\\\Sparse Chains\\\RIO_NP\sparse"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
from src.humobi.misc.generators import *
import os
from time import time
import json

df = TrajectoriesFrame(os.path.join(file_path, "RIO_NP.csv"))
df['labels'] = df.labels.astype(np.int64)
df = df.uloc(df.get_users()[:5])
# fname = open(os.path.join(save_path, 'rioNP_sparse_times.txt'), 'w')

comb_pred = []
for x in ['Q',None, 'L', 'Q','IW','IWS']:
    for y in ['IW', None, 'L', 'Q','IW','IWS']:
        for z in ['IWS', None, 'L', 'Q','IW','IWS']:
            for a in ['Q', None, 'L', 'Q','IW','IWS']:
                comb_pred.append((x, y, z, a))

test_size = .2
split_ratio = 1 - test_size
valtrain_frame, test_frame = split(df, split_ratio, 0)
val_frame = valtrain_frame.groupby(level=0).apply(lambda x: x.iloc[round(len(x) * .8):]).droplevel(1)
train_frame = valtrain_frame.groupby(level=0).apply(lambda x: x.iloc[:round(len(x) * .8)]).droplevel(1)
test_lengths = val_frame.groupby(level=0).apply(lambda x: x.shape[0])
lstart = time()
sparse_alg = sparse_wrapper_learn(train_frame, overreach=False, reverse=True, old=False,
                                  rolls=True, remove_subsets=False, reverse_overreach=False, jit=True,
                                  search_size=50, parallel=True)
lend = time()
ltime = lend - lstart
# fname.write("Learn time %s \n" % (str(ltime)))
for cp in comb_pred:
    tstart = time()
    forecast_df, pred_res, topk_res = sparse_wrapper_test(sparse_alg, val_frame, valtrain_frame, split_ratio, test_lengths,
                                                length_weights=cp[0], recency_weights=cp[1],
                                                org_length_weights=cp[2], org_recency_weights=cp[3], use_probs=False)
    tend = time()
    ttime = tend - tstart
    # fname.write("Pred time %s %s \n" % (str(cp), str(ttime)))
    pred_res.to_csv(os.path.join(save_path, 'rioNP_sparse_scores_%s_%s.csv' % ('SS50', cp)))
    forecast_df.to_csv(os.path.join(save_path, 'rioNP_sparse_predictions_%s_%s.csv' % ('SS50', cp)))
    with open(os.path.join(save_path,'rioNP_sparse_topk_%s_%s' % ('SS50',cp)),'w') as ff:
        json.dump(topk_res,ff)

# fname.close()
