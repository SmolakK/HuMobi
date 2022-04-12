import numpy as np
from numba import cuda, jit
import sys
sys.path.append("..")
from misc.generators import markovian_sequences_generator, deterministic_sequences_generator, random_sequences_generator, non_stationary_sequences_generator, exploratory_sequences_generator
import matplotlib.pyplot as plt
from math import ceil, floor
import time

import os
from structures.trajectory import TrajectoriesFrame
from measures.individual import random_predictability, unc_predictability, real_predictability, self_transitions
from tools.user_statistics import count_records
import sys
from predictors.wrapper import *
import numpy.ma as ma
# sys.path.append("..")
# direc = "D:\\Projekty\\bias\\london\\london_1H_204.33597178569417_1.csv"
# df = TrajectoriesFrame(direc, {'names': ['id','time','temp','lat','lon','labels','start','end','geometry'],
#                                    'delimiter': ',', 'skiprows': 1})

def normalize_chain(dicto):
	"""
	Normalizes dictionary values. Used for the Markov Chain normalization.
	:param dicto: dictionary to normalize
	:return: normalized dictionary
	"""
	total = 1 / float(np.sum(list(dicto.values())))
	for k, v in dicto.items():
		dicto[k] = v * total
	return dicto


def get_diags(a):
	"""
	Extracts all the diagonals from the matrix
	:param a: numpy array to process
	:return: a list of diagonals
	"""
	diags = [a.diagonal(i) for i in range(-a.shape[0] + 1, a.shape[1])]
	return [n.tolist() for n in diags]


def _repeatfinder_equally_sparse(s1, s2, variant):
	if len(s1) > len(s2):
		return None
	matrix = np.zeros((len(s1), len(s2)))  # prepare matrix for results
	for i in range(len(s1)):
		for j in range(len(s2)):
			if s1[i] == s2[j]:
				if i == 0 or j == 0:  # if matched symbols are at the start of the sequence
					matrix[i][j] += 1  # if symbols matched - add 1
				else:
					matrix[i][j] = 1  # if symbols matched - add 1
	s2_indi = (np.vstack(
		[np.arange(matrix.shape[1]) + 1 for x in range(matrix.shape[0])])) * matrix # convert matched 1's into indices
	s1_indi = (np.hstack(
		[np.expand_dims(np.arange(matrix.shape[0]),axis=1)+1 for x in range(matrix.shape[1])]))*matrix
	s2diags = get_diags(s2_indi)  # get all diagonals
	s1diags = get_diags(s1_indi)
	if sum([sum(x) for x in s2diags]) == 0:
		return None
	nonzero_s2 = [[y-1 for y in x if y != 0] for x in s2diags if sum(x) > 0]  # filter out empty lists
	nonzero_s1 = [[len(s1)-y+1 for y in x if y != 0] for x in s1diags if sum(x) > 0] # filter out empty lists
	# nonzero_s2 = [x for x in nonzero_s2 if len(x) >= 2]
	# nonzero_s1 = [x for x in nonzero_s1 if len(x) >= 2]
	matches = []
	if variant == 1:
		for x, y in zip(nonzero_s1, nonzero_s2):
			if y[-1]+x[-1] < len(s2):
				matched_pattern = [(int(z - y[-1]), s2[int(-(len(s2) - z))]) for z in y[:-1]]
				last_symbol = s2[int(y[-1])]
				matches.append((matched_pattern,last_symbol))
	elif variant == 2:
		for x, y in zip(nonzero_s1, nonzero_s2):
			if y[-1]+x[-1] < len(s2):
				matched_pattern = [(int(-w), s2[int(z)]) for z,w in zip(y,x)]
				next_symbol = s2[int(y[-1]+x[-1])]
				matches.append((matched_pattern,next_symbol))
	elif variant == 3:
		for x, y in zip(nonzero_s1, nonzero_s2):
			if y[-1]+x[-1] < len(s2):
				matched_pattern = tuple(y)
				next_symbol = s2[int(y[-1]+x[-1])]
				matches.append((matched_pattern,next_symbol))
	return matches

#TODO: Make it a model build (after x,pos y,pos there is-> through matching
def make_preds(variant = 3, app = 'lens'):
	final_res = []
	final_freq_res = []
	res = 0
	freq_res = 0
	cnt = -1
	for q in range(100):
		cnt += 1
		if cnt >= 100:
			break
		# seq = seqc[seqc.userid == cnt]
		seq = df.uloc(pd.unique(df.index.get_level_values(0))[cnt])
		S = seq.labels.values[:-1]
		scanthrough = {}
		prob_dict = {}
		for n in range(1,len(S)*2):
			cur_id = len(S)-n
			if cur_id > 0:
				lookback = S[cur_id:]
				search_space = S[:cur_id]
			elif cur_id < 0:
				lookback = S[:cur_id]
				search_space = S[cur_id:]
			out = _repeatfinder_equally_sparse(lookback,search_space,variant)
			if out:
				for ids, candidate in out:
					if candidate in scanthrough.keys():
						scanthrough[candidate].append(ids)
					else:
						scanthrough[candidate] = [ids]
		if variant < 3:
			for k,v in scanthrough.items():
				v = map(tuple,v)
				v = list(set(v))
				scanthrough[k] = v
			#MATCHING
			matches = {}
			for candidate, ids in scanthrough.items():
				for cases in ids:
					if not isinstance(cases[0], tuple):
						cases = [cases]
					partial_match = (S[[x[0] for x in cases]] == np.array([x[1] for x in cases]))
					if partial_match.any():
						if candidate in matches.keys():
							matches[candidate].append((cases,partial_match))
						else:
							matches[candidate] = [(cases,partial_match)]
			for candidate, match in matches.items():
				match_fil = [np.array(x[0])[x[1]] for x in match]
				recency = [abs(1/x[:,0]).sum() for x in match_fil]
				if app == 'lens_learn':
					prob_dict[candidate] = [a*b for a,b in zip([sum(x[1])/len(x[0]) for x in match],recency)]
				elif app == 'lengthofmatch_learn':
					prob_dict[candidate] = [len(x[:,0]) for x in match_fil]
				elif app == 'recency_learn':
					prob_dict[candidate] = [a*b for a,b in zip([x[:,1].shape[0]/len(y[0]) for x,y in zip(match_fil,match)],recency)]
				elif app == 'lengthrecency_learn':
					prob_dict[candidate] = [len(x[:, 0]) * y for x, y in zip(match_fil, recency)]
				elif app == 'fullmatcheslen_learn':
					prob_dict[candidate] = [len(x[0]) for x in match if x[1].all()]
				elif app == 'lenslengthofmatch_learn':
					prob_dict[candidate] = [len(x[0])*x[1]/len(x[0]) for x in match if x[1].all()]
			prob_dict = {k: sum((x)) for k, x in prob_dict.items()}
		else:
			if app == 'lens' or app == 'lensoflens':
				for k,v in scanthrough.items():
					v = map(tuple, v)
					v = list(set(v))
					scanthrough[k] = v
				if app == 'lens':
					prob_dict = {k:len((x)) for k,x in scanthrough.items()}
				elif app == 'lensoflens':
					prob_dict = {x:sum([len(z) for z in y]) for x,y in scanthrough.items()}
			else:
				if app == 'lensnonunique':
					prob_dict = {k: len((x)) for k, x in scanthrough.items()}
				elif app == 'lensoflensnonunique':
					prob_dict = {x: sum([len(z) for z in y]) for x, y in scanthrough.items()}
		try:
			normalize_chain(prob_dict)
			SMC = max(prob_dict, key=prob_dict.get)
		except:
			SMC = np.argmax(np.unique(S,return_counts=True)[1])
		freq = np.argmax(np.unique(S,return_counts=True)[1])
		if SMC == seq.labels.values[-1]:
			res += 1
		if freq == seq.labels.values[-1]:
			freq_res += 1
	final_res.append(res/cnt)
	final_freq_res.append(freq_res/cnt)
	print(p,res/cnt)
	print(p,freq_res/cnt)
	return final_res

sys.path.append("..")
direc = "D:\\Projekty\\bias\\london\\london_seq_33.430119654032815_1.csv"
df = TrajectoriesFrame(direc, {'names': ['id','datetime','lat','lon','geometry','labels','start','end'],
                                   'delimiter': ',', 'skiprows': 1})
for p in range(9,10):
	p /= 10
	# tests = 100
	seqc = df
	# seqc = random_sequences_generator(100,2,100)
# seqc = pd.read_csv('mark.csv')

	start = time.time()
	app='recency_learn'
	print(app)
	ys = make_preds(variant=2,app=app)
	end = time.time()
	print(end-start)
	plt.plot(np.arange(len(ys)),ys,label = app)
	start = time.time()
	app='lens_learn'
	print(app)
	ys = make_preds(variant=2,app=app)
	end = time.time()
	print(end-start)
	plt.plot(np.arange(len(ys)),ys,label = app)
	start = time.time()
	app='lengthofmatch_learn'
	print(app)
	ys = make_preds(variant=2,app=app)
	end = time.time()
	print(end-start)
	plt.plot(np.arange(len(ys)),ys,label = app)
	app='fullmatcheslen_learn'
	print(app)
	ys = make_preds(variant=2,app=app)
	end = time.time()
	print(end-start)
	plt.plot(np.arange(len(ys)),ys,label = app)

	start = time.time()
	app='lens'
	print(app)
	ys = make_preds(variant=3,app=app)
	end = time.time()
	print(end-start)
	plt.plot(np.arange(len(ys)),ys,label = app)
	start = time.time()
	app='lensoflens'
	print(app)
	ys = make_preds(variant=3,app=app)
	end = time.time()
	print(end-start)
	plt.plot(np.arange(len(ys)),ys,label = app)
	start = time.time()
	app='lensoflensnonunique'
	print(app)
	ys = make_preds(variant=3,app=app)
	end = time.time()
	print(end-start)
	plt.plot(np.arange(len(ys)),ys,label = app)

plt.legend(frameon=True)
plt.show()