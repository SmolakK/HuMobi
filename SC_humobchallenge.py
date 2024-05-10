import os
import pandas as pd
import tqdm
path = "D:\Projekty\Sparse Chains\paper_tests\HuMobChall"
file1 = "task2_dataset.csv"
file1_path = os.path.join(path,file1)
tdf = pd.read_csv(file1_path)
to_concat = []
grouped = tdf.groupby('uid')
for uid,df in tqdm.tqdm(grouped,total=len(grouped)):
    df['labels'] = df['x'].astype(str) + ',' + df['y'].astype(str)
    unique_coors = pd.DataFrame(pd.unique(df['labels']))
    sub_dict = unique_coors.to_dict()[0]
    sub_dict = {v:k for (k,v) in sub_dict.items()}
    df.astype({'labels': str})
    df['labels'] = df['labels'].map(sub_dict)
    to_concat.append(df)
tdf = pd.concat(to_concat)
tdf.set_index('uid')
tdf.to_csv(os.path.join(path,'task2_labeled.csv'))