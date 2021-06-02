import os
from structures.trajectory import TrajectoriesFrame
from measures.individual import random_predictability, unc_predictability, real_predictability, self_transitions
from tools.user_statistics import count_records
import sys

sys.path.append("..")
direc = "INPUT"  # INPUT DIR
direc2 = "OUTPUT"  # OUTPUT DIR

for r, d, f in os.walk(direc):
	for ffile in f:
		full_path = os.path.join(direc, ffile)
		loc, temp, size, eps = ffile.replace(".csv", "").split("_")  # READ NAMES
		df = TrajectoriesFrame(full_path, {
			'names': ['columns'], 'delimiter': ',', 'skiprows': 1})  # READ AGGREGATED DATA

		ffs = ffile.replace(".csv", "")  # FILENAME

		frac = df.groupby(level=0).apply(lambda x: (x.isnull().sum() / len(x)).iloc[0])  # FRACTION OF EMPTY RECORDS
		inter_level = set(frac[frac <= 0.6].index)  # FILTER q < .6
		if not (count_records(df) > 2).all():
			zero_level = set(
				count_records(df)[count_records(df) > 2].index)  # ONLY DATA HAVING AT LEAST TWO DATA POINTS
		try:
			inter_level = inter_level.intersection(zero_level)
		except:
			pass
		if len(inter_level) == 0:
			continue

		df.groupby(level=0).apply(lambda x: len(x.groupby(['labels']))).to_csv(
			os.path.join(direc2, ffs + "_stayregions.csv"))  # STAY REGIONS
		df.groupby(level=0).apply(lambda x: sum(1 - x.isna().iloc[:, 0])).to_csv(
			os.path.join(direc2, ffs + "_count.csv"))  # RECORDS COUNT
		self_transitions(df).to_csv(os.path.join(direc2, ffs + "_selfT.csv"))  # NUMBER OF SELF-TRANSITIONS

		real_pred, real_ent = real_predictability(df)  # CALCULATE THE ACTUAL ENTROPY AND PREDICTABILITY
		real_pred.to_csv(os.path.join(direc2, ffs + "_realpexp.csv"))
		real_ent.to_csv(os.path.join(direc2, ffs + "_realeexp.csv"))
		unc_pred, unc_ent = unc_predictability(df)  # CALCULATE THE UNCORRELATED ENTROPY AND PREDICTABILITY
		unc_pred.to_csv(os.path.join(direc2, ffs + "_uncp.csv"))
		unc_ent.to_csv(os.path.join(direc2, ffs + "_unce.csv"))
		ran_pred, ran_ent = random_predictability(df)  # CALCULATE THE RANDOM ENTROPY AND PREDICTABILITY
		ran_pred.to_csv(os.path.join(direc2, ffs + "_ranp.csv"))
		ran_ent.to_csv(os.path.join(direc2, ffs + "_rane.csv"))
