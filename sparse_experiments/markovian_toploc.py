top_path = """D:\\Projekty\\Sparse Chains\\markovian"""
from src.humobi.predictors.wrapper import *
from src.humobi.measures.individual import *
import os


df = TrajectoriesFrame(os.path.join(top_path,"markovian.csv"))
data_splitter = Splitter(split_ratio=.2, horizon=5, n_splits=1)
data_splitter.stride_data(df)
test_frame_X = data_splitter.test_frame_X
test_frame_Y = data_splitter.test_frame_Y
cv_data = data_splitter.cv_data

toploc_results = TopLoc(train_data=cv_data,
                        test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y]).predict()
toploc_results[0].to_csv(os.path.join(top_path,'predictions_toploc_markovian.csv'))
toploc_results[1].to_csv(os.path.join(top_path,'toploc_markovian.csv'))
