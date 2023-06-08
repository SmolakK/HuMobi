top_path = """D:\\Projekty\\Sparse Chains\\markovian"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
from src.humobi.misc.generators import *
from src.humobi.predictors.wrapper import *
from sklearn.ensemble import RandomForestClassifier
import os
from time import time
import json

df = TrajectoriesFrame(os.path.join(top_path,"markovian.csv"))
df = df.loc[:10]
test_size = .2
split_ratio = 1 - test_size
for H in range(20,21,2):
	data_splitter = Splitter(split_ratio=test_size, horizon=H, n_splits=5)
	data_splitter.stride_data(df)
	test_frame_X = data_splitter.test_frame_X
	test_frame_Y = data_splitter.test_frame_Y
	cv_data = data_splitter.cv_data
	# fname = open(os.path.join(top_path,'markovian_RF_times.txt'),'w')
	
	# SKLEARN CLASSIFICATION METHOD
	start = time()
	clf = RandomForestClassifier
	predic = SKLearnPred(algorithm=clf, training_data=data_splitter.cv_data,
	                     test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y],
	                     param_dist={'n_estimators': [x for x in range(500, 5000, 500)], 'max_depth': [None, 2, 4, 6, 8, 10, 12],
	                                 'min_samples_leaf': [x for x in range(1,10,1)], 'max_features': ['sqrt','log2'],
	                                 'class_weight': ['balanced','balanced_subsample',None], 'min_samples_split': [x for x in range(2,20,2)]},
	                     search_size=5, cv_size=1, parallel=False)
	predic.learn()
	end = time()
	print("RF LEARN TIME",end-start)
	start = time()
	predic.test()
	end = time()
	print("RF PRED TIME",end-start)
	rf_scores = predic.scores
	rf_scores.to_csv(os.path.join(top_path,str(H)+'_RF_scores.csv'))
