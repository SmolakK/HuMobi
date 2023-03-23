import numpy as np
import tqdm
from src.humobi.misc.utils import get_diags, normalize_chain, _equally_sparse_match


class Sparse(object):
	"""
	Sparse predictor
	"""

	def __init__(self, sequence):
		self._sequence = sequence
		self.model = self.build()

	def build(self):
		scanthrough = {}
		matches = []
		nexts = []
		for n in tqdm.tqdm(range(1, len(self._sequence)*2),total=len(self._sequence)*2-1):
			cur_id = len(self._sequence) - n
			if cur_id > 0:
				lookback = self._sequence[cur_id:]
				search_space = self._sequence[:cur_id]
			elif cur_id < 0:
				lookback = self._sequence[:cur_id]
				search_space = self._sequence[cur_id:]
			out = _equally_sparse_match(lookback, search_space)
			if out:
				matches.append(np.stack([x[0] for x in out]))
				nexts.append(np.stack([x[1] for x in out]))
		matches = np.vstack(matches)
		nexts = np.hstack(nexts)
		return matches,nexts

	def predict(self, context):
		matches = {}
		prob_dict = {}
		pad_size = self.model[0].shape[1] - context[0].shape[0]
		context = np.pad(context[0], (pad_size, 0))
		matches = np.sum((self.model[0] == context), axis=1) / self.model[0].shape[1]
		joined = np.vstack([matches.T, self.model[1]]).T
		joined = joined[joined[:,1].argsort()]
		spliter = np.unique(joined[:,1], return_index=True)
		joined = np.split(joined[:,0],spliter[1][1:])
		probs = [np.sum(x) for x in joined]
		SMC = spliter[0][np.argmax(probs)]
		# for candidate, ids in self.model.items():
		# 	cnts = ids[1]
		# 	ids = ids[0]
		# 	if not isinstance(ids[0], list):
		# 		ids = [ids]
		# 		cnts = [cnts]
		# 	for cases, each_count in zip(ids,cnts):
		# 		cases = [(x,y) for x,y in cases if abs(x) <= len(context)]
		# 		partial_match = (context[[int(x[0]) for x in cases]] == np.array([x[1] for x in cases]))
		# 		if partial_match.any():
		# 			if candidate in matches.keys():
		# 				matches[candidate].append((cases, partial_match,each_count)) #added count
		# 			else:
		# 				matches[candidate] = [(cases, partial_match,each_count)] #added count
		# for candidate, match in matches.items():
		# 	match_fil = [np.array(x[0])[x[1]] for x in match]
		# 	weights = [x**2 for x in range(len(self._sequence))]
		# 	recency = [abs(1 / x[:, 0].sum()) for x in match_fil]
		# 	# recency = [((len(self._sequence)+x[:, 0])/len(self._sequence)) for x in match_fil]
		# 	recency = [max([weights[int(z)]*z for z in (len(self._sequence)+x[:, 0])]) for x in match_fil]
		# 	all_counts = [x[2] for x in match]
		# 	recency = [x * y for x, y in zip(recency, all_counts)]
		# 	prob_dict[candidate] = [a * b for a, b in
		# 	                        zip([x[:, 1].shape[0] / len(y) for x, y in zip(match_fil, match)], recency)]
		# prob_dict = {k: sum((x)) for k, x in prob_dict.items()}
		# try:
		# 	normalize_chain(prob_dict)
		# 	SMC = max(prob_dict, key=prob_dict.get)
		# except ZeroDivisionError:
		# 	SMC = np.argmax(np.unique(context, return_counts=True)[1])
		# return SMC
