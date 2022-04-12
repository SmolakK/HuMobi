from src.humobi.misc.generators import *
from src.humobi.predictors.wrapper import *
from src.humobi.predictors.deep import *
from src.humobi.measures.individual import *
from sklearn.ensemble import RandomForestClassifier

# ALSO GENERATE SOME DATA
markovian_seq = markovian_sequences_generator(users=10, places=10, length=500, prob=.3)
random_seq = random_sequences_generator(users=10, places=10, length=100)
deter_seq = deterministic_sequences_generator(users=10, places=10, repeats=10)
ex_seq = exploratory_sequences_generator(users=10, places=10)
st_seq = self_transitions_sequences_generator(users=10, places=10, length=100)
non_st_seq = non_stationary_sequences_generator(users=10, places=10, states=5, length=100)

# NOW LET'S DO SOME PREDICTIONS
# DATA PREPARING (TEST-TRAIN SPLIT OF MULTIPLE TRAJECTORIES)
data_splitter = Splitter(markovian_seq, split_ratio=.2, horizon=1, n_splits=1)
test_frame_X = data_splitter.test_frame_X
test_frame_Y = data_splitter.test_frame_Y
cv_data = data_splitter.cv_data

# TEST-TRAIN-BASED METRICS
X = pd.concat([data_splitter.cv_data[0][0], data_splitter.cv_data[0][2]])
Y = test_frame_X
DR = repeatability_dense(X, Y)
SR = repeatability_sparse(X, Y)
ESR = repeatability_equally_sparse(X, Y)
GA = global_alignment(X, Y)
IGA = iterative_global_alignment(X, Y)

# LET'S MAKE PREDICTIONS

# TOPLOC ALGORITHM
toploc_results = TopLoc(train_data=cv_data,
                        test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y]).predict()

# DEEP LEARNING METHODS
GRU = DeepPred("GRU", markovian_seq, split_ratio=.8, folds=5, window_size=5, batch_size=50, embedding_dim=512,
                rnn_units=1024)
GRU.learn_predict()
GRU_score = GRU.scores

GRU2 = DeepPred("GRU2", markovian_seq, split_ratio=.8, folds=5, window_size=5, batch_size=50, embedding_dim=512,
                rnn_units=1024)
GRU2.learn_predict()
GRU2_score = GRU2.scores

# SKLEARN CLASSIFICATION METHOD
clf = RandomForestClassifier
predic = SKLearnPred(algorithm=clf, training_data=data_splitter.cv_data,
                     test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y],
                     param_dist={'n_estimators': [x for x in range(500, 5000, 500)], 'max_depth': [None, 2, 4, 6, 8]},
                     search_size=5, cv_size=5, parallel=True)
predic.learn()
predic.test()
rf_scores = predic.scores

# MARKOV CHAINS
MC1 = markov_wrapper(markovian_seq, split_ratio=.8, state_size=1, update=False, averaged=False, online=True)
MC2 = markov_wrapper(markovian_seq, split_ratio=.8, state_size=2, update=False, averaged=False, online=True)
MC2_offline = markov_wrapper(markovian_seq, split_ratio=.8, state_size=2, update=False, averaged=False, online=False)
MC2_updated = markov_wrapper(markovian_seq, split_ratio=.8, state_size=2, update=False, averaged=False, online=False)
