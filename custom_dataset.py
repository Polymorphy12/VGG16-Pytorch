import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

# df1 = pd.read_csv("birds.csv")
# df2 = pd.read_csv("class_dict.csv")
#
# print(df1)
# print(df2)
#
# def find_class(row):
#     cls = row['labels']
#     cls_num = df2.index[df2['class']==cls].tolist()
#     s = ''.join(map(str,cls_num))
#     cls_num = int(s)
#     return cls_num
#
# df1['class_index'] = df1.apply(find_class, axis=1)
#
# valid_dataset = df1.loc[df1['data set'] == 'valid']
# valid_dataset = valid_dataset.reset_index()
#
# train_dataset = df1.loc[df1['data set'] == 'train']
# train_dataset = train_dataset.reset_index()
#
# valid_dataset.to_csv("./valid.csv")
# train_dataset.to_csv("./train.csv")
# print(valid_dataset)


# df1 =pd.read_csv('train.csv')
# df2 =pd.read_csv('valid.csv')
#
# print(df1.iloc[0, 2])
# print(df1.iloc[0, 5])
# print(df2.iloc[0, 2])
# print(df2.iloc[0, 5])


class BirdsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 2])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 5]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)