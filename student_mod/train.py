from student_mod.barlow_twins import BarlowTwins, OnlineFineTuner
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from student_mod.config import *
import torch 
from pytorch_lightning import Trainer
from student_mod.model import model_student_mod
from student_mod.dataset import *

encoder_out_dim = 1000

model = BarlowTwins(
    encoder=model_student_mod,
    encoder_out_dim=encoder_out_dim,
    num_training_samples=len(train_dataset),
    batch_size=batch_size,
    z_dim=z_dim,
)

# # online_finetuner = OnlineFineTuner(encoder_output_dim=encoder_out_dim, num_classes=10)
# checkpoint_callback = ModelCheckpoint(every_n_epochs=10, save_top_k=-1, save_last=True)

# trainer = Trainer(
#     max_epochs=max_epochs,
#     accelerator="auto",
#     devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
#     callbacks=[
#                 # online_finetuner, 
#                 checkpoint_callback
#             ],
# )
# trainer.fit(model, train_loader, val_loader)