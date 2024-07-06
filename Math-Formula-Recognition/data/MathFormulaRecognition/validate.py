# 验证所有文件都存在，并给出统计数据

import os
import pandas as pd

CUR_DIR = os.path.dirname(__file__)
for csv_name,folder in zip(["groundtruth_latex.csv","groundtruth.csv"],["train_latex","train"]):
    data = pd.read_csv(CUR_DIR+"/"+csv_name)
    for path,formula in data.values:
        assert os.path.exists(f"{CUR_DIR}/{folder}/{path}"), f"{folder}/{path}"
    data['class'] = data['path'].apply(lambda x:x.split("/")[0])
    print(csv_name)
    print(data.shape[0])
    print(data.groupby('class').count()["path"])