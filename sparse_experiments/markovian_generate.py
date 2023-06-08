top_path = """D:\\Projekty\\Sparse Chains\\markovian"""
from src.humobi.misc.generators import *
from src.humobi.measures.individual import *
import os

markovian_seq = markovian_sequences_generator(users=100, places=[x for x in range(1,50)], length=[x for x in range(100,500,100)], prob=[.3,.5,.7,.9],
                                              return_params=True)
# markovian_seq[0].to_csv(os.path.join(top_path,"markovian.csv"))
# markovian_seq[1].to_csv(os.path.join(top_path,"markovian_params.csv"))