import pytorch_lightning as pl
from torchvision import transforms 

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchmetrics import Accuracy, Precision, Recall

from torchvision import transforms
from student_mod.model import get_mobile_vit
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import random
from student_mod.config import *

random.seed(43)
torch.manual_seed(43)


transform_train = A.Compose([
    A.Blur(),
    A.RandomContrast(),
    A.ColorJitter(),
    A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

transform_valid = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def trans_func_train(image):
    image = np.asarray(image)
    image_aug = transform_train(image=image)['image']
    return image_aug

def trans_func_valid(image):
    image = np.asarray(image)
    image_aug = transform_valid(image=image)['image']
    return image_aug


class StudentDataset():
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data 
        self.y_data = y_data
        self.transform = transform 

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx):
        img_np = self.x_data[idx]
        img = Image.fromarray(img_np).convert('L').convert('RGB')
  
        if self.transform:
            img = self.transform(img)

        return img



class PesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './'):
        super().__init__()
        self.batch_size = batch_size

        self.transform_train = trans_func_train

        self.transform_valid = trans_func_valid
        self.transform_test = trans_func_valid
        
        self.num_classes = 2

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = StudentDataset(np.load(data_dir_fine_x_train), 
                                             np.load(data_dir_fine_y_train), 
                                             transform=self.transform_train)
            self.data_val = StudentDataset(np.load(data_dir_fine_x_valid), 
                                             np.load(data_dir_fine_y_valid), 
                                             transform=self.transform_valid)
            self.data_test = StudentDataset(np.load(data_dir_fine_x_test), 
                                             np.load(data_dir_fine_y_test), 
                                             transform=self.transform_test)
            
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

from torchmetrics.functional import accuracy, precision, recall, f1_score
average = None

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model = get_mobile_vit()
        model.classifier.fc = nn.Linear(384, 2)

        self.model = model

        self.acc = accuracy
        self.pre = precision
        self.rec = recall
        
        self.all_preds = []
        self.all_labels = []
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        
        acc = self.acc(logits, y, num_classes=2)

        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        
        pred = logits.argmax(dim=1)
        
        self.all_preds.append(pred.to('cpu'))
        self.all_labels.append(y.to('cpu'))

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.all_preds,dim=0)
        all_labels = torch.cat(self.all_labels,dim=0)

        acc = accuracy(all_preds, all_labels)
        pre = precision(all_preds, all_labels, average=average, num_classes=2)
        rec = recall(all_preds, all_labels, average=average, num_classes=2)
        f1 = f1_score(all_preds, all_labels, average=average, num_classes=2)
        
        self.log('val_acc', acc)
        self.log('val_pre', pre[1])
        self.log('val_rec', rec[1])
        self.log('val_f1', f1[1])
        
        self.all_preds = []
        self.all_labels = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        
        
        pred = logits.argmax(dim=1)
        
        self.all_preds.append(pred.to('cpu'))
        self.all_labels.append(y.to('cpu'))
    
    def on_test_epoch_end(self):
        
        all_preds = torch.cat(self.all_preds,dim=0)
        all_labels = torch.cat(self.all_labels,dim=0)
        
        acc = accuracy(all_preds, all_labels)
        pre = precision(all_preds, all_labels, average=average, num_classes=2)
        rec = recall(all_preds, all_labels, average=average, num_classes=2)
        f1 = f1_score(all_preds, all_labels, average=average, num_classes=2)
        
        self.log('test_acc', acc)
        self.log('test_pre', pre[1])
        self.log('test_rec', rec[1])
        self.log('test_f1', f1[1])
        
        self.all_preds = []
        self.all_labels = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


dm = PesDataModule(batch_size=32)

model_lit = LitModel()

checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode='max')

from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="Student_mod", name="mobilevit", log_model="all")


# Initialize a trainer
trainer = pl.Trainer(max_epochs=100,
                     gpus=1, 
                     logger=wandb_logger,
                     callbacks=[
                                checkpoint_callback],
                     )

trainer.fit(model_lit, dm)

model_lit = LitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
trainer.test(model_lit, dm)

