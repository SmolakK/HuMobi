import os
from humobi.structures.trajectory import TrajectoriesFrame
from src.humobi.predictors.wrapper import *
import pickle

# FILE READ
top_path = r'D:\Projekty\Sparse Chains\paper_tests\markovian'
file_name = 'markovian.csv'
file_path = os.path.join(top_path,file_name)
df = TrajectoriesFrame(file_path)

times = []
# PREDICTION MC1
for state in range(1,6):
    MC1 = markov_wrapper(df, test_size=.2, state_size=state, update=False, online=True)
    predictions, acc, k_predictions, MC1_time = MC1
    predictions.to_csv(os.path.join(top_path,'predictions','MC{}.csv'.format(state)))
    with open(os.path.join(top_path,'predictions','k_MC{}.pkl'.format(state)), 'wb') as f:
        pickle.dump(k_predictions, f)
    times.append(MC1_time)
pd.DataFrame(times).to_csv(os.path.join(top_path,'predictions','times_MC.csv'.format(state)))
