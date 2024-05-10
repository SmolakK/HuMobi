from src.humobi.misc.generators import *
import os

top_path = """D:\\Projekty\\Sparse Chains\\paper_tests"""
# ALSO GENERATE
num_of_places = list(range(1,50))
lengths = list(range(500,5000,100))
probs = [x/100 for x in range(1,100)]
markovian_seq = markovian_sequences_generator(users=1000, places=num_of_places, length=lengths, prob=probs, return_params=True)
seqs, params = markovian_seq
params.columns = ['places','length','p']
params['predictability'] = params['p'] + (1-params['p']) / params['places']
seqs.to_csv(os.path.join(top_path,'markov','markovian.csv'))
params.to_csv(os.path.join(top_path,'markov','markovian_params.csv'))