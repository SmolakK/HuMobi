top_path = """D:\\Projekty\\Sparse Chains\\markovian"""
from src.humobi.misc.generators import *
from src.humobi.measures.individual import *
import os

random_seq = non_stationary_sequences_generator(users=100,  places=[x for x in range(1,50)], length=[x for x in range(100,500,100)],
                                                states=[x for x in range(1,10)],return_params=True)
# random_seq[0].to_csv(os.path.join(top_path,"nonstationary.csv"))
# random_seq[1].to_csv(os.path.join(top_path,"nonstationary_params.csv"))