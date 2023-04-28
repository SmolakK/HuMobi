from src.humobi.misc.generators import *
from src.humobi.predictors.wrapper import *
from src.humobi.predictors.deep import *
from src.humobi.measures.individual import *
from sklearn.ensemble import RandomForestClassifier


comb = []
for x in [None,'L','Q','IW','IWS']:
	for y in [None, 'L', 'Q', 'IW', 'IWS']:
		for z in [False,True]:
			comb.append((x,y,z))

# ALSO GENERATE SOME DATA
# markovian_seq = markovian_sequences_generator(users=20, places=[2,4,10], length=[50,70,100,150], prob=[.3,.5,.7,.9])
# markovian_seq = TrajectoriesFrame("markovian.csv")
markovian_seq = random_sequences_generator(users=10, places=[2,4,10], length=[50,70,100])
# markovian_seq = deterministic_sequences_generator(users=10, places=10, repeats=10)
# ex_seq = exploratory_sequences_generator(users=10, places=10)
# st_seq = self_transitions_sequences_generator(users=10, places=10, length=100)
# markovian_seq = non_stationary_sequences_generator(users=10, places=[3,5,7], states=[1,3,4,6], length=100)

# NOW LET'S DO SOME PREDICTIONS
# DATA PREPARING (TEST-TRAIN SPLIT OF MULTIPLE TRAJECTORIES)
# markovian_seq = TrajectoriesFrame("D:\\Projekty\\bias\\london\\london_1H_111.7572900082951_1.csv",{'names':['id','datetime','temp','lat','lon','labels','start','end','geometry'],"skiprows":1})
# markovian_seq = markovian_seq.uloc(markovian_seq.get_users()[:5]).fillna(0)
data_splitter = Splitter(split_ratio=.2, horizon=5, n_splits=2)
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

# SKLEARN CLASSIFICATION METHOD
# clf = RandomForestClassifier
# predic = SKLearnPred(algorithm=clf, training_data=data_splitter.cv_data,
#                      test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y],
#                      param_dist={'n_estimators': [x for x in range(500, 5000, 500)], 'max_depth': [None, 2, 4, 6, 8]},
#                      search_size=5, cv_size=2, parallel=False)
# predic.learn()
# predic.test()
# rf_scores = predic.scores

sparse_results = {}
for c in comb:
	sparse9 = sparse_wrapper(markovian_seq, test_size=.2, state_size=0, averaged=False, length_weights=c[0],
	                         recency_weights=c[1], use_probs=c[2],
	                         reverse=True, overreach=False, rolls = True, remove_subsets=True)
	sparse10 = sparse_wrapper(markovian_seq, test_size=.2, state_size=0, averaged=False, length_weights=c[0],
	                         recency_weights=c[1], use_probs=c[2],
	                         reverse=True, overreach=True, rolls = True, remove_subsets=True)
	sparse0 = sparse_wrapper(markovian_seq, test_size=.2, state_size=0, averaged=False, length_weights=c[0],
	                         recency_weights=c[1], use_probs=c[2],
	                         reverse=False, overreach=False, old=True)
	sparse5 = sparse_wrapper(markovian_seq,test_size=.2,state_size=0,averaged=False,length_weights=c[0],recency_weights=c[1],use_probs=True,
	                        reverse=False,overreach=False, rolls = True)
	sparse6 = sparse_wrapper(markovian_seq, test_size=.2, state_size=0, averaged=False, length_weights=c[0],
	                        recency_weights=c[1], use_probs=True,
	                        reverse=False, overreach=True, rolls = True)
	sparse7 = sparse_wrapper(markovian_seq, test_size=.2, state_size=0, averaged=False, length_weights=c[0],
	                         recency_weights=c[1], use_probs=True,
	                         reverse=True, overreach=True, rolls = True)
	sparse8 = sparse_wrapper(markovian_seq, test_size=.2, state_size=0, averaged=False, length_weights=c[0],
	                         recency_weights=c[1], use_probs=True,
	                         reverse=True, overreach=False, rolls = True)
	sparse1 = sparse_wrapper(markovian_seq,test_size=.2,state_size=0,averaged=False,length_weights=c[0],recency_weights=c[1],use_probs=c[2],
	                        reverse=False,overreach=False, rolls = True)
	sparse2 = sparse_wrapper(markovian_seq, test_size=.2, state_size=0, averaged=False, length_weights=c[0],
	                        recency_weights=c[1], use_probs=c[2],
	                        reverse=False, overreach=True, rolls = True)
	sparse3 = sparse_wrapper(markovian_seq, test_size=.2, state_size=0, averaged=False, length_weights=c[0],
	                         recency_weights=c[1], use_probs=c[2],
	                         reverse=True, overreach=True, rolls = True)
	sparse4 = sparse_wrapper(markovian_seq, test_size=.2, state_size=0, averaged=False, length_weights=c[0],
	                         recency_weights=c[1], use_probs=c[2],
	                         reverse=True, overreach=False, rolls = True)



	x = pd.concat([pd.DataFrame().from_dict(sparse0, orient='index'),
	               pd.DataFrame().from_dict(sparse1, orient='index'),
	               pd.DataFrame().from_dict(sparse2, orient='index'),
	              pd.DataFrame().from_dict(sparse3, orient='index'),
	               pd.DataFrame().from_dict(sparse4, orient='index'),
	               pd.DataFrame().from_dict(sparse5, orient='index'),
	               pd.DataFrame().from_dict(sparse6, orient='index'),
	               pd.DataFrame().from_dict(sparse7, orient='index'),
	               pd.DataFrame().from_dict(sparse8, orient='index'),
	               pd.DataFrame().from_dict(sparse9, orient='index'),
	               pd.DataFrame().from_dict(sparse10, orient='index'),
	               toploc_results[1]], axis=1)

	print(sparse1) #TODO: Separate train type and prediction type (prediction type based on the same train type)
	print(sparse2)
	print(sparse3)
	print(sparse4)

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
MC1 = markov_wrapper(markovian_seq, test_size=.2, state_size=1, update=False, online=True)
MC2 = markov_wrapper(markovian_seq, test_size=.2, state_size=2, update=False, online=True)
MC2_offline = markov_wrapper(markovian_seq, test_size=.2, state_size=2, update=False, online=False)
MC2_updated = markov_wrapper(markovian_seq, test_size=.2, state_size=2, update=False, online=False)


a