import os
from structures.trajectory import TrajectoriesFrame
from measures.individual import random_predictability, unc_predictability, real_predictability, self_transitions
from tools.user_statistics import count_records
import sys
from predictors.wrapper import *
from sklearn.ensemble import RandomForestClassifier

sys.path.append("..")
direc = "D:\\Projekty\\bias\\london\\london_6H_24.723129760223514_1.csv"
df = TrajectoriesFrame(direc, {'names': ['userid','datetime','temp','lat','lon','labels','start','end','geometry'],
                                   'delimiter': ',', 'skiprows': 1})
df = df.uloc(pd.unique(df.index.get_level_values(0))[0:5])
rp = real_predictability(df)
splt = Splitter(df,.8,10,1)
clf = RandomForestClassifier
predic = SKLearnPred(clf,splt.cv_data,[splt.test_frame_X,splt.test_frame_Y],{'n_estimators': [x for x in range(500,5000,500)],'max_depth':[None,2,4,6,8]}, search_size=5, cv_size=2)
predic.learn()
predic.test()
print(predic.scores)
y = sparse_wrapper(df,averaged=False)
x = markov_wrapper(df,state_size=1,averaged=True)
w = markov_wrapper(df,state_size=2,averaged=True)
z = markov_wrapper(df,state_size=3,averaged=True)
a = markov_wrapper(df,state_size=4,averaged=True)
b = markov_wrapper(df,state_size=5,averaged=True)
