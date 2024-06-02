# 直接将groundtruth_train.tsv分开来变为train/dev/test

import os
import pandas as pd
import random
from random import shuffle

def summary_dataset(data):
    data['class'] = data['path'].apply(lambda x: x.split("/")[0])
    print("数据数量",data.shape[0])
    print("种类分布")
    print(data.groupby('class').count()["path"])

def split(data, folder, train_dev_test=(0.8, 0.1, 0.1), seed=42):
    os.makedirs(folder,exist_ok=True)

    random.seed(seed)
    N = data.shape[0]
    idx = list(range(N))
    shuffle(idx)
    l = [0, 0, 0]
    l[0] = round(N * train_dev_test[0])
    l[1] = round(N * train_dev_test[1])
    l[2] = N - l[0] - l[1]

    train = data.loc[idx[0:l[0]], :].reset_index(drop=True)
    dev = data.loc[idx[l[0]:l[0] + l[1]], :].reset_index(drop=True)
    test = data.loc[idx[l[0] + l[1]:], :].reset_index(drop=True)

    train.to_csv(f"{folder}/train.csv", index=False)
    dev.to_csv(f"{folder}/dev.csv", index=False)
    test.to_csv(f"{folder}/test.csv", index=False)
    print('train')
    summary_dataset(train)
    print('dev')
    summary_dataset(dev)
    print('test')
    summary_dataset(test)


if __name__=="__main__":
    CUR_DIR = os.path.dirname(__file__)

    for name in ['groundtruth','groundtruth_latex']:
        data = pd.read_csv(f"{CUR_DIR}/data/MathFormulaRecognition/{name}.csv",
                           names=["path", "formula"])
        folder = f"{CUR_DIR}/data/MathFormulaRecognition/split"
        if "latex" in name:
            folder += "_latex"
        print('='*50)
        print(name)
        split(data, folder, seed=42)
