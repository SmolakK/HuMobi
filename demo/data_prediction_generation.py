from humobi.misc.generators import *
from humobi.predictors.wrapper import *
from humobi.predictors.deep import *
from humobi.measures.individual import *
from sklearn.ensemble import RandomForestClassifier

# ALSO GENERATE SOME DATA
# markovian_seq = markovian_sequences_generator(users=100, places=10, length=150, prob=[.3,.5,.7,.9])
# random_seq = random_sequences_generator(users=100, places=10, length=400)
# deter_seq = deterministic_sequences_generator(users=10, places=10, repeats=10)
# ex_seq = exploratory_sequences_generator(users=10, places=10)
# st_seq = self_transitions_sequences_generator(users=10, places=10, length=100)
# non_st_seq = non_stationary_sequences_generator(users=10, places=10, states=5, length=100)

markovian_seq = TrajectoriesFrame("mark_seq.csv")
# fpath = """D:\papier2\london_1H_45.20353656360243_1.csv"""
# markovian_seq = TrajectoriesFrame(fpath, {'names': ['id','time','temp','lat','lon','labels','start','end','geometry'],'skiprows':1})
# markovian_seq = markovian_seq.uloc([x for x in markovian_seq.get_users()][:40])
# NOW LET'S DO SOME PREDICTIONS
# DATA PREPARING (TEST-TRAIN SPLIT OF MULTIPLE TRAJECTORIES)
data_splitter = Splitter(markovian_seq, split_ratio=.2, horizon=5, n_splits=2)
test_frame_X = data_splitter.test_frame_X
test_frame_Y = data_splitter.test_frame_Y
cv_data = data_splitter.cv_data

# TEST-TRAIN-BASED METRICS
X = pd.concat([data_splitter.cv_data[0][0], data_splitter.cv_data[0][2]])
Y = test_frame_X
DR = repeatability_dense(train_frame=X, test_frame=Y)
SR = repeatability_sparse(train_frame=X, test_frame=Y)
ESR = repeatability_equally_sparse(train_frame=X, test_frame=Y)
GA = global_alignment(train_frame=X, test_frame=Y)
IGA = iterative_global_alignment(train_frame=X, test_frame=Y)

# LET'S MAKE PREDICTIONS
a = sparse_wrapper(markovian_seq,test_size=.2,state_size=0,averaged=False)
pd.Series(a).to_csv('c.csv')
oppaaa
# MARKOV CHAINS
MC1 = markov_wrapper(markovian_seq, test_size=.2, state_size=1, update=False, online=True)
MC1[1].to_csv('MC1.csv')
MC2 = markov_wrapper(markovian_seq, test_size=.2, state_size=2, update=False, online=True)
MC2[1].to_csv('MC2.csv')
MC2_offline = markov_wrapper(markovian_seq, test_size=.2, state_size=2, update=False, online=False)
MC2_offline[1].to_csv('MC2off.csv')
MC2_updated = markov_wrapper(markovian_seq, test_size=.2, state_size=2, update=False, online=False)
MC2_updated[1].to_csv("MC2up.csv")

# TOPLOC ALGORITHM
toploc_results = TopLoc(train_data=cv_data,
                        test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y]).predict()
toploc_results[1].to_csv('toploc.csv')



# SKLEARN CLASSIFICATION METHOD
clf = RandomForestClassifier
predic = SKLearnPred(algorithm=clf, training_data=data_splitter.cv_data,
                     test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y],
                     param_dist={'n_estimators': [x for x in range(500, 5000, 500)], 'max_depth': [None, 2, 4, 6, 8]},
                     search_size=1, cv_size=1, parallel=False)
predic.learn()
predic.test()
rf_scores = predic.scores
rf_scores.to_csv('rf.csv')

# # DEEP LEARNING METHODS
GRU = DeepPred("GRU", markovian_seq, test_size=.2, folds=5, window_size=10, batch_size=1, embedding_dim=512,
                rnn_units=1024)
GRU.learn_predict()
GRU_score = GRU.scores
GRU_score.to_csv("GRU.csv")

GRU = DeepPred("GRU", markovian_seq, test_size=.2, folds=1, window_size=10, batch_size=1, embedding_dim=512,
                rnn_units=1024)
GRU.learn_predict()
GRU_score2 = GRU.scores
GRU_score2.to_csv("GRU2.csv")


GRU = DeepPred("GRU", markovian_seq, test_size=.2, folds=10, window_size=10, batch_size=1, embedding_dim=512,
                rnn_units=1024)
GRU.learn_predict()
GRU_score3 = GRU.scores
GRU_score3.to_csv("GRU3.csv")


# GRU2 = DeepPred("GRU2", markovian_seq, test_size=.2, folds=5, window_size=5, batch_size=50, embedding_dim=512,
#                 rnn_units=1024)
# GRU2.learn_predict()
# GRU2_score = GRU2.scores