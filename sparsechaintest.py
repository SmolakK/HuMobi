from src.humobi.misc.generators import *
from src.humobi.predictors.wrapper import *
from src.humobi.predictors.deep import *
from src.humobi.measures.individual import *
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score
from time import time


SEARCH_SIZE = 20
top_path = """D:\\Projekty\\Sparse Chains\\quick"""
# ALSO GENERATE SOME DATA
# markovian_seq = markovian_sequences_generator(users=20, places=[2,4,10], length=[1000,500,700,1500], prob=[.3,.5,.7,.9])
# markovian_seq.to_csv("markovian.csv")
markovian_seq = TrajectoriesFrame("markovian.csv")
markovian_seq = markovian_seq.uloc(markovian_seq.get_users()[:10])
# markovian_seq = random_sequences_generator(users=10, places=[2,4,10], length=[50,70,100])
# markovian_seq = deterministic_sequences_generator(users=10, places=10, repeats=10)
# ex_seq = exploratory_sequences_generator(users=10, places=10)
# st_seq = self_transitions_sequences_generator(users=10, places=10, length=100)
# markovian_seq = non_stationary_sequences_generator(users=10, places=[3,5,7], states=[1,3,4,6], length=100)

# NOW LET'S DO SOME PREDICTIONS
# DATA PREPARING (TEST-TRAIN SPLIT OF MULTIPLE TRAJECTORIES)
# markovian_seq = TrajectoriesFrame("D:\\Projekty\\bias\\london\\london_1H_111.7572900082951_1.csv",{'names':['id','datetime','temp','lat','lon','labels','start','end','geometry'],"skiprows":1})
# markovian_seq = markovian_seq.uloc(markovian_seq.get_users()[1:6]).fillna(0)
data_splitter = Splitter(split_ratio=.2, horizon=5, n_splits=1)
data_splitter.stride_data(markovian_seq)
test_frame_X = data_splitter.test_frame_X
test_frame_Y = data_splitter.test_frame_Y
cv_data = data_splitter.cv_data

# TEST-TRAIN-BASED METRICS
# X = pd.concat([data_splitter.cv_data[0][0], data_splitter.cv_data[0][2]])
# Y = test_frame_X
# DR = repeatability_dense(train_frame=X, test_frame=Y)
# SR = repeatability_sparse(train_frame=X, test_frame=Y)
# ESR = repeatability_equally_sparse(train_frame=X, test_frame=Y)
# GA = global_alignment(train_frame=X, test_frame=Y)
# IGA = iterative_global_alignment(train_frame=X, test_frame=Y)

# LET'S MAKE PREDICTIONS

# TOPLOC ALGORITHM
toploc_results = TopLoc(train_data=cv_data,
                        test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y]).predict()

toploc_results[1].to_csv(os.path.join(top_path,'toploc.csv'))

# SKLEARN CLASSIFICATION METHOD
start = time()
clf = RandomForestClassifier
predic = SKLearnPred(algorithm=clf, training_data=data_splitter.cv_data,
                     test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y],
                     param_dist={'n_estimators': [x for x in range(500, 5000, 500)], 'max_depth': [None, 2, 4, 6, 8]},
                     search_size=1, cv_size=1, parallel=False)
predic.learn()
end = time()
print("RF LEARN TIME",end-start)
start = time()
predic.test()
end = time()
print("RF PRED TIME",end-start)
rf_scores = predic.scores
rf_scores.to_csv(os.path.join(top_path,'rf.csv'))

sparse_results = {}

#EXPERIMENTAL SPARSES
overreach = True
reverse = True
rolls = True
reverse_overreach = True  # the only changeable
jit = True

test_size = .2
train, test = [x.droplevel(0) for x in split(markovian_seq,1-test_size,0)]
cv_data = expanding_split(train,5)
save_path = r"D:\Projekty\Sparse Chains\quick"

for SEARCH_SIZE in [20]:
    start = time()
    best_combos = sparse_wrapper(trajectories_frame = cv_data, search_size = SEARCH_SIZE)
    end = time()
    accs = {}
    topks = {}
    top3s = {}
    top5s = {}
    top10s = {}
    f1s = {}
    times = {}
    for uid in pd.unique(train.index.get_level_values(0)):
        uid_model = Sparse(overreach=overreach, reverse=reverse, rolls=rolls,
               reverse_overreach=reverse_overreach,
               search_size=SEARCH_SIZE)
        train_frame_X = train.loc[uid]
        test_frame_X = test.loc[uid]
        uid_model.fit(train.loc[uid].values.ravel())
        forecast, topk = predict_with_hyperparameters(train_frame_X, test_frame_X, cur_model = uid_model, jit = True, use_probs = False, **best_combos[uid])
        accuracy_score = sum(forecast == test_frame_X.values.ravel())/len(forecast)
        lbls = pd.unique(test_frame_X)
        f1 = f1_score(test_frame_X.values.ravel(),forecast,labels=lbls,average='macro')
        topk_width = max(map(len,topk.values()))
        topk_stacked = [np.pad(x,(0,topk_width-x.shape[0])) for x in topk.values()]
        topk_stacked = np.stack(topk_stacked)
        topk_sorted = np.argsort(topk_stacked,axis=1)
        top3_acc = (test_frame_X.values.ravel()[...,None] == topk_sorted[:,-3:]).any(axis=1).sum()/topk_sorted.shape[0]
        top5_acc = (test_frame_X.values.ravel()[..., None] == topk_sorted[:, -5:]).any(axis=1).sum()/topk_sorted.shape[0]
        top10_acc = (test_frame_X.values.ravel()[..., None] == topk_sorted[:, -10:]).any(axis=1).sum()/topk_sorted.shape[0]
        f1s[uid] = f1
        top3s[uid] = top3_acc
        top5s[uid] = top5_acc
        top10s[uid] = top10_acc
        accs[uid] = accuracy_score
        topks[uid] = topk
        times[uid] = end-start
    pd.DataFrame().from_dict(top3s,orient='index').to_csv(os.path.join(save_path,str(SEARCH_SIZE)+'top3.csv'))
    pd.DataFrame().from_dict(top5s,orient='index').to_csv(os.path.join(save_path,str(SEARCH_SIZE)+'top5.csv'))
    pd.DataFrame().from_dict(top10s,orient='index').to_csv(os.path.join(save_path,str(SEARCH_SIZE)+'top10.csv'))
    pd.DataFrame().from_dict(accs,orient='index').to_csv(os.path.join(save_path,str(SEARCH_SIZE)+'accs.csv'))
    pd.DataFrame().from_dict(f1s,orient='index').to_csv(os.path.join(save_path,str(SEARCH_SIZE)+'f1s.csv'))
    pd.DataFrame().from_dict(times, orient='index').to_csv(os.path.join(save_path, str(SEARCH_SIZE) + 'times.csv'))

# DEEP LEARNING METHODS
GRU = DeepPred("GRU", markovian_seq, test_size=.2, folds=5, window_size=10, batch_size=10, embedding_dim=512,
                rnn_units=1024)
GRU.learn_predict()
GRU_score = GRU.scores
#
GRU2 = DeepPred("GRU", markovian_seq, test_size=.2, folds=1, window_size=10, batch_size=10, embedding_dim=512,
                rnn_units=1024)
GRU2.learn_predict()
GRU_score2 = GRU2.scores
#
GRU2 = DeepPred("GRU2", markovian_seq, test_size=.2, folds=5, window_size=5, batch_size=50, embedding_dim=512,
                rnn_units=1024)
GRU2.learn_predict()
GRU2_score = GRU2.scores

# MARKOV CHAINS
MC1 = markov_wrapper(markovian_seq, test_size=.2, state_size=1, update=False, online=True)
MC2 = markov_wrapper(markovian_seq, test_size=.2, state_size=2, update=False, online=True)
MC2_offline = markov_wrapper(markovian_seq, test_size=.2, state_size=2, update=False, online=False)
MC2_updated = markov_wrapper(markovian_seq, test_size=.2, state_size=2, update=False, online=False)
a