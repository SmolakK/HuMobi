import os
from humobi.structures.trajectory import TrajectoriesFrame
from src.humobi.predictors.wrapper import *
import pickle
from sklearn.ensemble import RandomForestClassifier

# FILE READ
top_path = r'D:\Projekty\Sparse Chains\paper_tests\HuMobChall'
file_name = 'task1_labeled.csv'
file_path = os.path.join(top_path,file_name)
df = pd.read_csv(file_path)
users = pd.unique(df.uid)
top10 = users[:10]
df = df[df.uid.isin(top10)]
df['datetime'] = pd.to_datetime(df['d'], unit='D') + pd.to_timedelta(df['t'] - 1, unit='h')
df = df[['uid','datetime','labels']]
df.columns = ['user_id','datetime','labels']
df = df.set_index(['user_id','datetime'],drop=False)

for h in [5,10,20,30,50,100,200,300,500]:
    data_splitter = Splitter(split_ratio=.2, horizon=h, n_splits=5)
    data_splitter.stride_data(df)
    test_frame_X = data_splitter.test_frame_X
    test_frame_Y = data_splitter.test_frame_Y
    cv_data = data_splitter.cv_data

    times = []
    start = time()
    clf = RandomForestClassifier
    predic = SKLearnPred(algorithm=clf, training_data=data_splitter.cv_data,
                         test_data=[data_splitter.test_frame_X, data_splitter.test_frame_Y],
                         param_dist={'n_estimators': [x for x in range(500, 5000, 500)], 'max_depth': [None, 2, 4, 6, 8],
                                     'min_samples_split':[2,4,6,8],'min_samples_leaf':[1,2,4,6,8]},
                         search_size=1, cv_size=1, parallel=True) # cv depends on data
    predic.learn()
    end = time()
    learntime = end - start
    start = time()
    predic.test()
    end = time()
    predtime = end - start
    predic.predictions.to_csv(os.path.join(top_path,'predictions','RF{}.csv'.format(h)))
    with open(os.path.join(top_path, 'predictions', 'k_RF{}.pkl'.format(h)), 'wb') as f:
        pickle.dump(predic.predictions_proba, f)
    pd.DataFrame([learntime,predtime]).to_csv(os.path.join(top_path,'predictions','RF{}_times.csv'.format(h)))
