import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.humobi.structures.trajectory import TrajectoriesFrame
from src.humobi.measures.individual import *
from src.humobi.tools.user_statistics import fraction_of_empty_records, count_records, count_records_per_time_frame, \
	user_trajectories_duration, consecutive_record
from src.humobi.predictors.wrapper import Splitter
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import numpy as np

from scipy.stats import spearmanr
import pandas as pd


def calculate_pvalues(df):
	df = df.dropna()._get_numeric_data()
	dfcols = pd.DataFrame(columns=df.columns)
	pvalues = dfcols.transpose().join(dfcols, how='outer')
	for r in df.columns:
		for c in df.columns:
			pvalues[r][c] = round(spearmanr(df[r], df[c])[1], 4)
	return pvalues


def logarithmic_one(x, a, b, c):
	return a * np.log(x * b) + c


def line_one(x, a, b):
	return a * x + b


def expon_one(x, a, b, c):
	return a * x ** b + c


def logarithmic_mixed_two(x, a, b, c, d, e):
	return a * x[0, :] + c * np.log(x[1, :]) + e + b * x[0, :] * np.log(x[1, :])


def logarithmic_full_two(x, a, b, c, d, e):
	return a * np.log(x[0, :]) + c * np.log(x[1, :]) + e + b * np.log(x[0, :]) * np.log(x[1, :])


def line_two(x, a, b, c):
	return a * x[0, :] + b * x[1, :] + c


def expon_two(x, a, b, c, d, e):
	return a * x[0, :] ** b + c * x[1, :] ** d + e


def estimate(func, xs, ys):
	try:
		popt, pcov = curve_fit(func, xs, ys)
		r2 = r2_score(ys, func(xs, *popt))
		if xs.ndim > 1:
			r2 = 1 - (1 - r2) * ((xs.shape[1] - 1) / (xs.shape[1] - xs.shape[0] - 1))
		return r2
	except:
		return 0


dirr = """D:\\papier2\\results_tables"""
related_dir = """D:\\papier2\\final"""
# pred_list = {}
# eq5_list = {}
# smax_list = {}
# eq5_stat_list = {}
# smax_stat_list = {}
# eq5_stat_new_list = {}
# smax_stat_new_list = {}
# eq5_dense_list = {}
# smax_dense_list = {}
# for r, d, f in os.walk(dirr):
# 	for ff in f:
# 		if 'markovian_r' in ff or 'nonstationary_r' in ff or 'random_r' in ff:
# 			fpath = os.path.join(dirr, ff)
# 			df = pd.read_csv(fpath, index_col=0)
# 			beta = df.mean()[['deep', 'RF']].idxmax()
# 			striped_ff = ff.replace("_result", "")
# 			full_path = os.path.join(related_dir, striped_ff)
# 			related_df = pd.read_csv(full_path, index_col=0, header=0)
# 			related_df['user_id'] = related_df['userid']
# 			related_df.index = related_df['user_id']
# 			df['stationarity'] = stationarity(related_df)
#
# 			# PRED tests
# 			ys = df[beta]
# 			xs = df['pred']
# 			pred_res = {}
# 			pred_res['line'] = estimate(line_one, xs, ys)
# 			pred_res['expo'] = estimate(expon_one, xs, ys)
# 			pred_res['log'] = estimate(logarithmic_one, xs, ys)
# 			pred_res = pd.DataFrame().from_dict(pred_res, orient='index')
# 			pred_list[striped_ff] = pd.DataFrame(pred_res)
#
# 			# EQ5 tests
# 			ys = df[beta]
# 			xs = df['eq5']
# 			eq5_res = {}
# 			eq5_res['line'] = estimate(line_one, xs, ys)
# 			eq5_res['expo'] = estimate(expon_one, xs, ys)
# 			eq5_res['log'] = estimate(logarithmic_one, xs, ys)
# 			eq5_res = pd.DataFrame().from_dict(eq5_res, orient='index')
# 			eq5_list[striped_ff] = pd.DataFrame(eq5_res)
#
# 			# SMAX tests
# 			ys = df[beta]
# 			xs = df['s_max']
# 			smax_res = {}
# 			smax_res['line'] = estimate(line_one, xs, ys)
# 			smax_res['expo'] = estimate(expon_one, xs, ys)
# 			smax_res['log'] = estimate(logarithmic_one, xs, ys)
# 			smax_res = pd.DataFrame().from_dict(smax_res, orient='index')
# 			smax_list[striped_ff] = pd.DataFrame(smax_res)
#
# 			# (eq5, stat),
# 			eq5_stat = {}
# 			xs = df[['eq5', 'stat']].values.T
# 			eq5_stat['line'] = estimate(line_two, xs, ys)
# 			eq5_stat['expo'] = estimate(expon_two, xs, ys)
# 			eq5_stat['logf'] = estimate(logarithmic_full_two, xs, ys)
# 			eq5_stat['log'] = estimate(logarithmic_mixed_two, xs, ys)
# 			eq5_stat = pd.DataFrame().from_dict(eq5_stat, orient='index')
# 			eq5_stat_list[striped_ff] = pd.DataFrame(eq5_stat)
#
# 			# (s_max, stat),
# 			smax_stat = {}
# 			xs = df[['s_max', 'stat']].values.T
# 			smax_stat['line'] = estimate(line_two, xs, ys)
# 			smax_stat['expo'] = estimate(expon_two, xs, ys)
# 			smax_stat['logf'] = estimate(logarithmic_full_two, xs, ys)
# 			smax_stat['log'] = estimate(logarithmic_mixed_two, xs, ys)
# 			smax_stat = pd.DataFrame().from_dict(smax_stat, orient='index')
# 			smax_stat_list[striped_ff] = pd.DataFrame(smax_stat)
#
# 			# (eq5, stat_new),
# 			eq5_stat_new = {}
# 			xs = df[['eq5', 'stationarity']].values.T
# 			eq5_stat_new['line'] = estimate(line_two, xs, ys)
# 			eq5_stat_new['expo'] = estimate(expon_two, xs, ys)
# 			eq5_stat_new['logf'] = estimate(logarithmic_full_two, xs, ys)
# 			eq5_stat_new['log'] = estimate(logarithmic_mixed_two, xs, ys)
# 			eq5_stat_new = pd.DataFrame().from_dict(eq5_stat_new, orient='index')
# 			eq5_stat_new_list[striped_ff] = pd.DataFrame(eq5_stat_new)
#
# 			# (s_max, stat_new),
# 			smax_stat_new = {}
# 			xs = df[['s_max', 'stationarity']].values.T
# 			smax_stat_new['line'] = estimate(line_two, xs, ys)
# 			smax_stat_new['expo'] = estimate(expon_two, xs, ys)
# 			smax_stat_new['logf'] = estimate(logarithmic_full_two, xs, ys)
# 			smax_stat_new['log'] = estimate(logarithmic_mixed_two, xs, ys)
# 			smax_stat_new = pd.DataFrame().from_dict(smax_stat_new, orient='index')
# 			smax_stat_new_list[striped_ff] = pd.DataFrame(smax_stat_new)
#
# 			# (eq5, max_dense),
# 			eq5_dense = {}
# 			xs = df[['eq5', 'max_dense']].values.T
# 			eq5_dense['line'] = estimate(line_two, xs, ys)
# 			eq5_dense['expo'] = estimate(expon_two, xs, ys)
# 			eq5_dense['logf'] = estimate(logarithmic_full_two, xs, ys)
# 			eq5_dense['log'] = estimate(logarithmic_mixed_two, xs, ys)
# 			eq5_dense = pd.DataFrame().from_dict(eq5_dense, orient='index')
# 			eq5_dense_list[striped_ff] = pd.DataFrame(eq5_dense)
#
# 			# (s_max, stat_new),
# 			smax_dense = {}
# 			xs = df[['s_max', 'max_dense']].values.T
# 			smax_dense['line'] = estimate(line_two, xs, ys)
# 			smax_dense['expo'] = estimate(expon_two, xs, ys)
# 			smax_dense['logf'] = estimate(logarithmic_full_two, xs, ys)
# 			smax_dense['log'] = estimate(logarithmic_mixed_two, xs, ys)
# 			smax_dense = pd.DataFrame().from_dict(smax_dense, orient='index')
# 			smax_dense_list[striped_ff] = pd.DataFrame(smax_dense)
#
# synth_conc = {}
# for k in eq5_list.keys():
# 	synth_conc[k] = pd.concat(
# 		[eq5_list[k], smax_list[k], eq5_stat_list[k], smax_stat_list[k], eq5_stat_new_list[k], smax_stat_new_list[k],
# 		 eq5_dense_list[k], smax_dense_list[k], pred_list[k]], axis=1).T

# related_dir = """D:\\papier2\\final\\nmdc"""
# eq5_list = {}
# smax_list = {}
# eq5_stat_list = {}
# smax_stat_list = {}
# eq5_stat_new_list = {}
# smax_stat_new_list = {}
# eq5_dense_list = {}
# smax_dense_list = {}
# for r, d, f in os.walk(dirr):
# 	for ff in f:
# 		if 'nmdc_' in ff:
# 			fpath = os.path.join(dirr, ff)
# 			df = pd.read_csv(fpath, index_col=0)
# 			beta = df.mean()[['deep', 'RF']].idxmax()
# 			if 'seq' in ff:
# 				striped_ff = ff.replace("result_", "")
# 			else:
# 				striped_ff = ff.replace("result_", "").replace("nmdc", 'ndmc')
#
# 			y = df[beta]
# 			xs = df['s_max']
#
# 			# EQ5 tests
# 			ys = df[beta]
# 			xs = df['eq5']
# 			eq5_res = {}
# 			eq5_res['line'] = estimate(line_one, xs, ys)
# 			eq5_res['expo'] = estimate(expon_one, xs, ys)
# 			eq5_res['log'] = estimate(logarithmic_one, xs, ys)
# 			eq5_res = pd.DataFrame().from_dict(eq5_res, orient='index')
# 			eq5_list[striped_ff] = pd.DataFrame(eq5_res)
#
# 			# SMAX tests
# 			ys = df[beta]
# 			xs = df['s_max']
# 			smax_res = {}
# 			smax_res['line'] = estimate(line_one, xs, ys)
# 			smax_res['expo'] = estimate(expon_one, xs, ys)
# 			smax_res['log'] = estimate(logarithmic_one, xs, ys)
# 			smax_res = pd.DataFrame().from_dict(smax_res, orient='index')
# 			smax_list[striped_ff] = pd.DataFrame(smax_res)
#
# 			# (eq5, stat),
# 			eq5_stat = {}
# 			xs = df[['eq5', 'stat']].values.T
# 			eq5_stat['line'] = estimate(line_two, xs, ys)
# 			eq5_stat['expo'] = estimate(expon_two, xs, ys)
# 			eq5_stat['logf'] = estimate(logarithmic_full_two, xs, ys)
# 			eq5_stat['log'] = estimate(logarithmic_mixed_two, xs, ys)
# 			eq5_stat = pd.DataFrame().from_dict(eq5_stat, orient='index')
# 			eq5_stat_list[striped_ff] = pd.DataFrame(eq5_stat)
#
# 			# (s_max, stat),
# 			smax_stat = {}
# 			xs = df[['s_max', 'stat']].values.T
# 			smax_stat['line'] = estimate(line_two, xs, ys)
# 			smax_stat['expo'] = estimate(expon_two, xs, ys)
# 			smax_stat['logf'] = estimate(logarithmic_full_two, xs, ys)
# 			smax_stat['log'] = estimate(logarithmic_mixed_two, xs, ys)
# 			smax_stat = pd.DataFrame().from_dict(smax_stat, orient='index')
# 			smax_stat_list[striped_ff] = pd.DataFrame(smax_stat)
#
# 			# (eq5, stat_new),
# 			eq5_stat_new = {}
# 			xs = df[['eq5', 'stationarity']].values.T
# 			eq5_stat_new['line'] = estimate(line_two, xs, ys)
# 			eq5_stat_new['expo'] = estimate(expon_two, xs, ys)
# 			eq5_stat_new['logf'] = estimate(logarithmic_full_two, xs, ys)
# 			eq5_stat_new['log'] = estimate(logarithmic_mixed_two, xs, ys)
# 			eq5_stat_new = pd.DataFrame().from_dict(eq5_stat_new, orient='index')
# 			eq5_stat_new_list[striped_ff] = pd.DataFrame(eq5_stat_new)
#
# 			# (s_max, stat_new),
# 			smax_stat_new = {}
# 			xs = df[['s_max', 'stationarity']].values.T
# 			smax_stat_new['line'] = estimate(line_two, xs, ys)
# 			smax_stat_new['expo'] = estimate(expon_two, xs, ys)
# 			smax_stat_new['logf'] = estimate(logarithmic_full_two, xs, ys)
# 			smax_stat_new['log'] = estimate(logarithmic_mixed_two, xs, ys)
# 			smax_stat_new = pd.DataFrame().from_dict(smax_stat_new, orient='index')
# 			smax_stat_new_list[striped_ff] = pd.DataFrame(smax_stat_new)
#
# 			# (eq5, max_dense),
# 			eq5_dense = {}
# 			xs = df[['eq5', 'max_dense']].values.T
# 			eq5_dense['line'] = estimate(line_two, xs, ys)
# 			eq5_dense['expo'] = estimate(expon_two, xs, ys)
# 			eq5_dense['logf'] = estimate(logarithmic_full_two, xs, ys)
# 			eq5_dense['log'] = estimate(logarithmic_mixed_two, xs, ys)
# 			eq5_dense = pd.DataFrame().from_dict(eq5_dense, orient='index')
# 			eq5_dense_list[striped_ff] = pd.DataFrame(eq5_dense)
#
# 			# (s_max, stat_new),
# 			smax_dense = {}
# 			xs = df[['s_max', 'max_dense']].values.T
# 			smax_dense['line'] = estimate(line_two, xs, ys)
# 			smax_dense['expo'] = estimate(expon_two, xs, ys)
# 			smax_dense['logf'] = estimate(logarithmic_full_two, xs, ys)
# 			smax_dense['log'] = estimate(logarithmic_mixed_two, xs, ys)
# 			smax_dense = pd.DataFrame().from_dict(smax_dense, orient='index')
# 			smax_dense_list[striped_ff] = pd.DataFrame(smax_dense)
#
# nmdc_conc = {}
# for k in eq5_list.keys():
# 	nmdc_conc[k] = pd.concat(
# 		[eq5_list[k], smax_list[k], eq5_stat_list[k], smax_stat_list[k], eq5_stat_new_list[k],
# 		 smax_stat_new_list[k], eq5_dense_list[k], smax_dense_list[k]], axis=1).T


related_dir = """D:\\Projekty\\bias\\london"""
eq5_list = {}
smax_list = {}
eq5_stat_list = {}
smax_stat_list = {}
eq5_stat_new_list = {}
smax_stat_new_list = {}
eq5_dense_list = {}
smax_dense_list = {}
pred_list = {}
pred_corr = {}
for r, d, f in os.walk(dirr):
	for ff in f:
		if 'london_' in ff:
			fpath = os.path.join(dirr, ff)
			print(ff)
			df = pd.read_csv(fpath, index_col=0)
			beta = df.mean()[['deep', 'RF']].idxmax()
			striped_ff = ff.replace("result_", "")
			related_df = TrajectoriesFrame(os.path.join(related_dir,striped_ff),
			                               {'names':['user_id','time','temp','lat','lon','labels','start','end','geometry'],'skiprows':1})
			related_df = to_labels(related_df)
			spliter = Splitter(related_df,.8)
			train = pd.concat([spliter.cv_data[0][0],spliter.cv_data[0][2]],axis=0).sort_index()
			test = spliter.test_frame_X
			df['IGA'] = iterative_global_alignment(train,test)
			df['GA'] = global_alignment(train,test)
			df['ESR'] = repeatability_equally_sparse(train,test)

			# PRED tests
			ys = df[beta]
			xs = df['pred']
			pred_res = {}
			pred_res['line'] = estimate(line_one, xs, ys)
			pred_res['expo'] = estimate(expon_one, xs, ys)
			pred_res['log'] = estimate(logarithmic_one, xs, ys)
			pred_res = pd.DataFrame().from_dict(pred_res, orient='index')
			pred_list[striped_ff] = pd.DataFrame(pred_res)
			pred_corr[striped_ff] = df.corr('spearman')[beta]['pred']

			# EQ5 tests
			ys = df[beta]
			xs = df['eq5']
			eq5_res = {}
			eq5_res['line'] = estimate(line_one, xs, ys)
			eq5_res['expo'] = estimate(expon_one, xs, ys)
			eq5_res['log'] = estimate(logarithmic_one, xs, ys)
			eq5_res = pd.DataFrame().from_dict(eq5_res, orient='index')
			eq5_list[striped_ff] = pd.DataFrame(eq5_res)

			# SMAX tests
			ys = df[beta]
			xs = df['s_max']
			smax_res = {}
			smax_res['line'] = estimate(line_one, xs, ys)
			smax_res['expo'] = estimate(expon_one, xs, ys)
			smax_res['log'] = estimate(logarithmic_one, xs, ys)
			smax_res = pd.DataFrame().from_dict(smax_res, orient='index')
			smax_list[striped_ff] = pd.DataFrame(smax_res)

			# (eq5, stat),
			eq5_stat = {}
			xs = df[['eq5', 'stat']].values.T
			eq5_stat['line'] = estimate(line_two, xs, ys)
			eq5_stat['expo'] = estimate(expon_two, xs, ys)
			eq5_stat['logf'] = estimate(logarithmic_full_two, xs, ys)
			eq5_stat['log'] = estimate(logarithmic_mixed_two, xs, ys)
			eq5_stat = pd.DataFrame().from_dict(eq5_stat, orient='index')
			eq5_stat_list[striped_ff] = pd.DataFrame(eq5_stat)

			# (s_max, stat),
			smax_stat = {}
			xs = df[['s_max', 'stat']].values.T
			smax_stat['line'] = estimate(line_two, xs, ys)
			smax_stat['expo'] = estimate(expon_two, xs, ys)
			smax_stat['logf'] = estimate(logarithmic_full_two, xs, ys)
			smax_stat['log'] = estimate(logarithmic_mixed_two, xs, ys)
			smax_stat = pd.DataFrame().from_dict(smax_stat, orient='index')
			smax_stat_list[striped_ff] = pd.DataFrame(smax_stat)

			# (eq5, stat_new),
			eq5_stat_new = {}
			xs = df[['eq5', 'stationarity']].values.T
			eq5_stat_new['line'] = estimate(line_two, xs, ys)
			eq5_stat_new['expo'] = estimate(expon_two, xs, ys)
			eq5_stat_new['logf'] = estimate(logarithmic_full_two, xs, ys)
			eq5_stat_new['log'] = estimate(logarithmic_mixed_two, xs, ys)
			eq5_stat_new = pd.DataFrame().from_dict(eq5_stat_new, orient='index')
			eq5_stat_new_list[striped_ff] = pd.DataFrame(eq5_stat_new)

			# (s_max, stat_new),
			smax_stat_new = {}
			xs = df[['s_max', 'stationarity']].values.T
			smax_stat_new['line'] = estimate(line_two, xs, ys)
			smax_stat_new['expo'] = estimate(expon_two, xs, ys)
			smax_stat_new['logf'] = estimate(logarithmic_full_two, xs, ys)
			smax_stat_new['log'] = estimate(logarithmic_mixed_two, xs, ys)
			smax_stat_new = pd.DataFrame().from_dict(smax_stat_new, orient='index')
			smax_stat_new_list[striped_ff] = pd.DataFrame(smax_stat_new)

			# (eq5, max_dense),
			eq5_dense = {}
			xs = df[['eq5', 'max_dense']].values.T
			eq5_dense['line'] = estimate(line_two, xs, ys)
			eq5_dense['expo'] = estimate(expon_two, xs, ys)
			eq5_dense['logf'] = estimate(logarithmic_full_two, xs, ys)
			eq5_dense['log'] = estimate(logarithmic_mixed_two, xs, ys)
			eq5_dense = pd.DataFrame().from_dict(eq5_dense, orient='index')
			eq5_dense_list[striped_ff] = pd.DataFrame(eq5_dense)

			# (s_max, stat_new),
			smax_dense = {}
			xs = df[['s_max', 'max_dense']].values.T
			smax_dense['line'] = estimate(line_two, xs, ys)
			smax_dense['expo'] = estimate(expon_two, xs, ys)
			smax_dense['logf'] = estimate(logarithmic_full_two, xs, ys)
			smax_dense['log'] = estimate(logarithmic_mixed_two, xs, ys)
			smax_dense = pd.DataFrame().from_dict(smax_dense, orient='index')
			smax_dense_list[striped_ff] = pd.DataFrame(smax_dense)

london_conc = {}
for k in eq5_list.keys():
	london_conc[k] = pd.concat(
		[eq5_list[k], smax_list[k], eq5_stat_list[k], smax_stat_list[k], eq5_stat_new_list[k],
		 smax_stat_new_list[k], eq5_dense_list[k], smax_dense_list[k], pred_list[k]], axis=1).T

a
