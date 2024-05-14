import os

import pandas as pd
from humobi.structures.trajectory import TrajectoriesFrame
from src.humobi.predictors.wrapper import *
import pickle

# FILE READ
top_path = r'D:\Projekty\Sparse Chains\paper_tests\markovian'
file_name = 'markovian.csv'
file_path = os.path.join(top_path,file_name)
df = TrajectoriesFrame(file_path)
df = df.loc[:5]

times = []
# PREDICTION SC
# EXPERIMENTAL SPARSES
overreach = True
reverse = True
rolls = True
reverse_overreach = True  # the only changeable
jit = True

test_size = .2
train, test = [x.droplevel(0) for x in split(df,1-test_size,0)]
cv_data = expanding_split(train,2)
save_path = r"D:\Projekty\Sparse Chains\paper_tests\markovian\predictions"

just_for_tests = {}
for SEARCH_SIZE in [40]:
    print(SEARCH_SIZE)
    start = time()
    best_combos = sparse_wrapper(trajectories_frame=cv_data, search_size=SEARCH_SIZE) # selects best params for each user
    end = time()
    us_res = []
    for uid in pd.unique(train.index.get_level_values(0)):
        uid_model = Sparse(overreach=overreach, reverse=reverse, rolls=rolls,
                           reverse_overreach=reverse_overreach,
                           search_size=SEARCH_SIZE)
        train_frame_X = train.loc[uid]
        test_frame_X = test.loc[uid]
        uid_model.fit(train.loc[uid].values.ravel())
        forecast, topk = predict_with_hyperparameters(train_frame_X, test_frame_X, cur_model = uid_model, jit = True, use_probs = False, **best_combos[uid])
        accuracy_score = sum(forecast == test_frame_X.values.ravel()) / len(forecast)
        us_res.append(accuracy_score)
    just_for_tests[SEARCH_SIZE] = us_res
pd.DataFrame.from_dict(just_for_tests,orient='index').to_csv(r'D:\SC_markovian.csv')
