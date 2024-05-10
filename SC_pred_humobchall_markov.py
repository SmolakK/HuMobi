import os
from humobi.structures.trajectory import TrajectoriesFrame
from src.humobi.predictors.wrapper import *
import pickle
import pandas as pd
from numpy import random

# FILE READ
top_path = r'D:\Projekty\Sparse Chains\paper_tests\HuMobChall'
file_name = 'task2_labeled.csv'
file_path = os.path.join(top_path,file_name)
df = pd.read_csv(file_path)
users = pd.unique(df.uid)
top10 = users[:5000]
df = df[df.uid.isin(top10)]
df = df.set_index(['uid','d','t'])

times = []
# PREDICTION MC1
for state in range(1,6):
    start = time()
    MC1 = markov_wrapper(df, test_size=.2, state_size=state, update=False, online=True)
    predictions, acc, k_predictions, MC1_time = MC1
    predictions.to_csv(os.path.join(top_path,'predictions2','MC{}.csv'.format(state)))
    with open(os.path.join(top_path,'predictions2','k_MC{}.pkl'.format(state)), 'wb') as f:
        pickle.dump(k_predictions, f)
    times.append(MC1_time)
pd.DataFrame(times).to_csv(os.path.join(top_path,'predictions2','times_MC.csv'.format(state)))