import torch
# import lightly
from torchvision import datasets, transforms
from student_mod.transform import BarlowTwinsTransform
from student_mod.config import *
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
import os
from natsort import natsorted
from PIL import Image



# collate_fn = lightly.data.SimCLRCollateFunction(
#     input_size=img_size
# )

# dataset_train_moco = lightly.data.LightlyDataset(
#     input_dir=path_to_train,
# )

# dataloader_train_moco = torch.utils.data.DataLoader(
#     dataset_train_moco,
#     batch_size=batch_size,
#     shuffle=True,
#     collate_fn=collate_fn,
#     drop_last=True,
#     num_workers=num_workers
# )


class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images
            transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        image_names = os.listdir(root_dir)

        self.root_dir = root_dir
        self.transform = transform 
        self.image_names = natsorted(image_names)

    def __len__(self): 
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get the path to the image 
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        img = img.resize(max_size=300)
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img


def normalization():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    return normalize


train_transform = BarlowTwinsTransform(
    train=True, input_height=224, gaussian_blur=False, jitter_strength=0.5, normalize=normalization()
)

train_dataset = CelebADataset(root_dir=data_dir_train, transform=train_transform)

# val_transform = BarlowTwinsTransform(
#     train=False, input_height=32, gaussian_blur=False, jitter_strength=0.5, normalize=normalization()
# )

# val_dataset = CelebADataset(root_dir="data/CelebA/img_align_celeba", train=False, download=True, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
