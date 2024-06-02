import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class MFRDataset(Dataset):
    def __init__(self, csv_path, pic_folder):
        self.csv_path = csv_path
        data = pd.read_csv(csv_path)
        self.data = []  # images
        for idx in tqdm(range(data.shape[0])):
            row = data.loc[idx,:]
            img = Image.open(f"{pic_folder}/{row['path']}").convert('RGB')
            self.data.append([img, row['formula']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__=="__main__":
    CUR_DIR = os.path.dirname(__file__)
    dev = MFRDataset(csv_path=f"{CUR_DIR}/data/MathFormulaRecognition/split_latex/dev.csv", # train
                     pic_folder=f"{CUR_DIR}/data/MathFormulaRecognition/train_latex")
    dev[0][0].show()
    print(dev[0][1])