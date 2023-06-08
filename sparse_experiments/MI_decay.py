
from src.humobi.predictors.wrapper import *
from src.humobi.misc.generators import *
from src.humobi.measures.individual import *
import os
from time import time
from sklearn.metrics import mutual_info_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
import json

# top_path = """D:\\Projekty\\Sparse Chains\\markovian"""
# df = TrajectoriesFrame(os.path.join(top_path,"markovian.csv")).loc[20:30]
df = TrajectoriesFrame("D:\\Projekty\\bias\\london\\london_1H_111.7572900082951_1.csv",{'names':['id','datetime','temp','lat','lon','labels','start','end','geometry'],"skiprows":1})
df = df.uloc(df.get_users()[:5]).fillna(0)
df['labels'] = df.labels.astype(np.int64)
#TODO: not corrected MI

def grassberger(total,counts):
	return np.log2(total) - np.sum([x * digamma(x) for x in counts])

def calc_MI(seq,distance,grasberger=True):
	unique_elements, counts = np.unique(seq, return_counts=True)
	total_elements = len(seq)
	probabilities = counts / total_elements
	marginal_entropies = -probabilities * np.log2(probabilities)
	marginal_entropies_dict = dict(zip(unique_elements, marginal_entropies))
	pairs = []
	for pairn in range(len(seq)-distance):
		pairs.append((seq[pairn],seq[pairn+distance]))
	pairs, pairscounts = np.unique(pairs,return_counts=True,axis=0)
	joint_probs = pairscounts/np.sum(pairscounts)
	mi_count = 0
	for x in range(len(pairs)):
		a = pairs[x][0]
		b = pairs[x][1]
		mi_count += marginal_entropies_dict[a] + marginal_entropies_dict[b] - joint_probs[x]
	return mi_count

top_path = """D:\\Projekty\\Sparse Chains\\markovian\\search_space.json"""
searchspace_reads = json.load(open(top_path))

mi_usr = {}
for uid,val in df.groupby(level=0):
	mi = []
	for n in range(val.shape[0]):
		mi.append(calc_MI(val.labels.values,n))
	mi_usr[uid] = mi
mi_usr