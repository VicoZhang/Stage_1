import torch
from torch.utils.data import Dataset, random_split
import os
from PIL import Image
from torchvision import transforms


class ReadData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        self.transforms = transforms.ToTensor()

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        img = self.transforms(img)
        label = eval(self.label_dir)  # 这里有一个小坑
        if label == 1:
            label = 0
        elif label == 4:
            label = 1
        elif label == 7:
            label = 2
        elif label == 8:
            label = 3
        else:
            label = 'wrong'
        label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.img_path)


data_dir = '../Data/DataSet/T1/'
type1_label_dir = '1'
type2_label_dir = '4'
type3_label_dir = '7'
type4_label_dir = '8'

type1_dataset = ReadData(data_dir, type1_label_dir)
type2_dataset = ReadData(data_dir, type2_label_dir)
type3_dataset = ReadData(data_dir, type3_label_dir)
type4_dataset = ReadData(data_dir, type4_label_dir)

len_train = 1500
len_test = 300
train_type1_dataset, test_type1_dataset = random_split(
    dataset=type1_dataset,
    lengths=[len_train, len_test],
    generator=torch.Generator().manual_seed(0)
)
train_type2_dataset, test_type2_dataset = random_split(
    dataset=type2_dataset,
    lengths=[len_train, len_test],
    generator=torch.Generator().manual_seed(0)
)
train_type3_dataset, test_type3_dataset = random_split(
    dataset=type3_dataset,
    lengths=[len_train, len_test],
    generator=torch.Generator().manual_seed(0)
)
train_type4_dataset, test_type4_dataset = random_split(
    dataset=type4_dataset,
    lengths=[len_train, len_test],
    generator=torch.Generator().manual_seed(0)
)

train_dataset = train_type1_dataset + train_type2_dataset + train_type3_dataset + train_type4_dataset
test_dataset = test_type1_dataset + test_type2_dataset + test_type3_dataset + test_type4_dataset
