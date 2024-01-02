from src.humobi.misc.generators import *
from src.humobi.predictors.wrapper import *
from src.humobi.predictors.deep import *
from src.humobi.measures.individual import *
import os
from sklearn.ensemble import RandomForestClassifier

SEARCH_SIZE = 30
# top_path = """D:\\Projekty\\Sparse Chains\\markovian"""
# ALSO GENERATE SOME DATA
markovian_seq = markovian_sequences_generator(users=5, places=[2,4,10], length=[100,500], prob=[.3,.5,.7,.9])
# markovian_seq.to_csv("markovian.csv")
# markovian_seq = TrajectoriesFrame("markovian.csv")
# markovian_seq = random_sequences_generator(users=10, places=[2,4,10], length=[50,70,100])
# markovian_seq = deterministic_sequences_generator(users=10, places=10, repeats=10)
# ex_seq = exploratory_sequences_generator(users=10, places=10)
# st_seq = self_transitions_sequences_generator(users=10, places=10, length=100)
# markovian_seq = non_stationary_sequences_generator(users=10, places=[3,5,7], states=[1,3,4,6], length=100)

# NOW LET'S DO SOME PREDICTIONS
# DATA PREPARING (TEST-TRAIN SPLIT OF MULTIPLE TRAJECTORIES)
# markovian_seq = TrajectoriesFrame("D:\\Projekty\\bias\\london\\london_1H_111.7572900082951_1.csv",{'names':['id','datetime','temp','lat','lon','labels','start','end','geometry'],"skiprows":1})
# markovian_seq = markovian_seq.uloc(markovian_seq.get_users()[1:6]).fillna(0)
data_splitter = Splitter(split_ratio=.3, horizon=5, n_splits=1)
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
# toploc_results = TopLoc(train_data=cv_data,
#                         test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y]).predict()
#
# toploc_results[1].to_csv(os.path.join(top_path,'toploc.csv'))
#
# # SKLEARN CLASSIFICATION METHOD
# start = time()
# clf = RandomForestClassifier
# predic = SKLearnPred(algorithm=clf, training_data=data_splitter.cv_data,
#                      test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y],
#                      param_dist={'n_estimators': [x for x in range(500, 5000, 500)], 'max_depth': [None, 2, 4, 6, 8]},
#                      search_size=1, cv_size=1, parallel=False)
# predic.learn()
# end = time()
# print("RF LEARN TIME",end-start)
# start = time()
# predic.test()
# end = time()
# print("RF PRED TIME",end-start)
# rf_scores = predic.scores
# rf_scores.to_csv(os.path.join(top_path,'rf.csv'))

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

best_combos = sparse_wrapper(trajectories_frame = cv_data, search_size = SEARCH_SIZE)
accs = {}
topks = {}
for uid in pd.unique(train.index.get_level_values(0)):
    uid_model = Sparse(overreach=overreach, reverse=reverse, rolls=rolls,
           reverse_overreach=reverse_overreach,
           search_size=SEARCH_SIZE)
    train_frame_X = train.loc[uid]
    test_frame_X = test.loc[uid]
    uid_model.fit(train.loc[uid].values.ravel())
    forecast, topk = predict_with_hyperparameters(train_frame_X, test_frame_X, cur_model = uid_model, jit = True, use_probs = False, **best_combos[uid])
    accuracy_score = sum(forecast == test_frame_X.values.ravel())/len(forecast)
    accs[uid] = accuracy_score
    topks[uid] = topk
accs

# #FIRST - SPARSE LEARN TYPE, SECOND - WEIGHTING TYPE
# sparse_alg = sparse_wrapper_learn(train_frame, overreach=True, reverse=True, old = False,
#                                   rolls=True, remove_subsets=True, reverse_overreach=True,
#                                   search_size=10)
# alg_weights = {}
# for c in comb:
# 	pred_res = sparse_wrapper_test(sparse_alg, test_frame, markovian_seq, split_ratio, test_lengths,
# 	                    length_weights=c[0],recency_weights=c[1],use_probs=c[2])
# 	alg_weights[c] = pred_res


# # DEEP LEARNING METHODS
# GRU = DeepPred("GRU", markovian_seq, test_size=.2, folds=5, window_size=10, batch_size=10, embedding_dim=512,
#                 rnn_units=1024)
# GRU.learn_predict()
# GRU_score = GRU.scores
#
# GRU2 = DeepPred("GRU", markovian_seq, test_size=.2, folds=1, window_size=10, batch_size=10, embedding_dim=512,
#                 rnn_units=1024)
# GRU2.learn_predict()
# GRU_score2 = GRU2.scores

# GRU2 = DeepPred("GRU2", markovian_seq, test_size=.2, folds=5, window_size=5, batch_size=50, embedding_dim=512,
#                 rnn_units=1024)
# GRU2.learn_predict()
# GRU2_score = GRU2.scores

# MARKOV CHAINS
# MC1 = markov_wrapper(markovian_seq, test_size=.2, state_size=1, update=False, online=True)
# MC2 = markov_wrapper(markovian_seq, test_size=.2, state_size=2, update=False, online=True)
# MC2_offline = markov_wrapper(markovian_seq, test_size=.2, state_size=2, update=False, online=False)
# MC2_updated = markov_wrapper(markovian_seq, test_size=.2, state_size=2, update=False, online=False)

markovian_test_file.close()