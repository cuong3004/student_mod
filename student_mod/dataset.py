import torch
# import lightly
from torchvision import datasets, transforms
from student_mod.transform import BarlowTwinsTransform
from student_mod.config import *
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data import random_split
from natsort import natsorted
from PIL import Image
from PIL.Image import Resampling
import pytorch_lightning as pl
import numpy as np

class CelebAStudentDataModule(pl.LightningDataModule):
    def __init__(self, 
        data_train_loader,
        data_valid_loader,
        data_train_fine_loader,
        data_valid_fine_loader,
        ) -> None:
        super().__init__()
        self.data_train_loader = data_train_loader
        self.data_valid_loader = data_valid_loader
        self.data_train_fine_loader = data_train_fine_loader
        self.data_valid_fine_loader = data_valid_fine_loader

    def train_dataloader(self):
        return [self.data_train_loader, self.data_train_fine_loader]

    def val_dataloader(self):
        return [self.data_valid_loader, self.data_valid_fine_loader]

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        image_names = os.listdir(root_dir)

        self.root_dir = root_dir
        self.transform = transform 
        self.image_names = natsorted(image_names)

    def __len__(self): 
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img.thumbnail((84, 84), Resampling.LANCZOS)
        if self.transform:
            img = self.transform(img)

        return img

class StudentDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data 
        self.y_data = y_data
        self.transform = transform 

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.root_dir, self.image_names[idx])
        img_np = self.x_data[idx]
        img = Image.fromarray(img_np).convert('L').convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img


def normalization():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    return normalize

def grayScale(x):
    img = x.convert("L")
    img = img.convert("RGB")
    return img


train_transform = BarlowTwinsTransform(
    train=True, input_height=64, gaussian_blur=False, jitter_strength=0.5, normalize=normalization()
)


train_transform_finetune = transforms.Compose(
                [   
                    transforms.Resize((64,64)),
                    transforms.RandomHorizontalFlip(),
                    transforms.Blur(),
                    transforms.ToTensor(),
                ]
            )
valid_transform_finetune = transforms.Compose(
                [   
                    transforms.Resize((64,64)),
                    transforms.ToTensor(),
                ]
            )

train_dataset = CelebADataset(root_dir=data_dir_train, transform=train_transform)
train_len = int(len(train_dataset)*0.9)
valid_len = len(train_dataset) - train_len
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_len, valid_len], generator=torch.Generator().manual_seed(42))


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)




# val_transform = BarlowTwinsTransform(
#     train=False, input_height=32, gaussian_blur=False, jitter_strength=0.5, normalize=normalization()
# )

# val_dataset = CelebADataset(root_dir="data/CelebA/img_align_celeba", train=False, download=True, transform=train_transform)
x_data_train = np.load(data_dir_fine_x_train)
y_data_train = np.load(data_dir_fine_y_train)

x_data_valid = np.load(data_dir_fine_x_valid)
y_data_valid = np.load(data_dir_fine_y_valid)

x_data_test = np.load(data_dir_fine_x_test)
y_data_test = np.load(data_dir_fine_y_test)

######
# train_dataset_fine = StudentDataset(x_data_train, y_data_train, transform=train_transform_finetune)
# valid_dataset_fine = StudentDataset(x_data_valid, y_data_valid, transform=valid_transform_finetune)
# test_dataset_fine = StudentDataset(x_data_test, y_data_test, transform=valid_transform_finetune)

# train_dataset_fine = datasets.ImageFolder(root=data_dir_fine_train, transform=train_transform_finetune)
# valid_dataset_fine = datasets.ImageFolder(root=data_dir_fine_valid, transform=valid_transform_finetune)

# train_dataset_fine = datasets.ImageFolder(root=data_dir_fine_train, transform=train_transform_finetune)
# valid_dataset_fine = datasets.ImageFolder(root=data_dir_fine_valid, transform=valid_transform_finetune)

#####
# train_fine_loader = DataLoader(train_dataset_fine, batch_size=32, shuffle=True, num_workers=num_workers, drop_last=True)
# valid_fine_loader = DataLoader(valid_dataset_fine, batch_size=32, shuffle=False, num_workers=num_workers, drop_last=True)
# valid_fine_loader = DataLoader(valid_dataset_fine, batch_size=32, shuffle=False, num_workers=num_workers, drop_last=True)

# dm = CelebAStudentDataModule(train_loader, valid_loader, train_fine_loader, valid_fine_loader)