top_path = """D:\\Projekty\\Sparse Chains\\markovian"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
from src.humobi.misc.generators import *
from src.humobi.predictors.wrapper import *
from sklearn.ensemble import RandomForestClassifier
import os
from time import time
import json

df = TrajectoriesFrame("D:\\Projekty\\bias\\london\\london_seq_111.7572900082951_1.csv",
                       {'names':['id','datetime','lat','lon','geometry','labels','start','end'],"skiprows":1})
df['labels'] = df.labels.astype(np.int64)
df = df.uloc(df.get_users()[:100])
fname = open(os.path.join(top_path,'humanNP_RF_times.txt'),'w')

for usr in range(0,len(df.get_users())-2,2):
	# SKLEARN CLASSIFICATION METHOD
	users = df.get_users()
	df_part = df.loc[users[usr:usr+2]]
	test_size = .2
	split_ratio = 1 - test_size
	data_splitter = Splitter(split_ratio=test_size, horizon=10, n_splits=5)
	data_splitter.stride_data(df_part)
	test_frame_X = data_splitter.test_frame_X
	test_frame_Y = data_splitter.test_frame_Y
	cv_data = data_splitter.cv_data
	start = time()
	clf = RandomForestClassifier
	predic = SKLearnPred(algorithm=clf, training_data=data_splitter.cv_data,
	                     test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y],
	                     param_dist={'n_estimators': [x for x in range(500, 5000, 500)],
	                                 'max_depth': [None, 2, 4, 6, 8, 10, 12],
	                                 'min_samples_leaf': [x for x in range(1, 10, 1)], 'max_features': ['sqrt', 'log2'],
	                                 'class_weight': ['balanced', 'balanced_subsample', None],
	                                 'min_samples_split': [x for x in range(2, 20, 2)]},
	                     search_size=10, parallel=True)
	predic.learn()
	end = time()
	print("RF LEARN TIME", end - start)
	fname.write("Learn time %s \n" % str((end-start)/2))
	start = time()
	predic.test()
	end = time()
	print("RF PRED TIME", end - start)
	fname.write("PRED time %s \n" % str((end-start)/2))
	rf_predictions = predic.predictions
	rf_predictions.columns = ['predictions','y_set']
	rf_scores = predic.scores
	rf_scores.to_csv(os.path.join(top_path,'RF_part','%s_humanNP_RF_scores.csv' % usr))
	rf_predictions.to_csv(os.path.join(top_path, 'RF_part', '%s_humanNP_RF_predictions.csv' % usr))
fname.close()