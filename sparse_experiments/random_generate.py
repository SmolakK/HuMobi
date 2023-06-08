top_path = """D:\\Projekty\\Sparse Chains\\markovian"""
from src.humobi.misc.generators import *
from src.humobi.measures.individual import *
import os

random_seq = random_sequences_generator(users=100, places=[x for x in range(1,50)], length=[x for x in range(100,500,100)],
                                              return_params=True)
# random_seq[0].to_csv(os.path.join(top_path,"random.csv"))
# random_seq[1].to_csv(os.path.join(top_path,"random_params.csv"))