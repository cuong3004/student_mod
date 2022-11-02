from student_mod.barlow_twins import BarlowTwins, OnlineFineTuner
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from student_mod.config import *
import torch 
from pytorch_lightning import Trainer
from student_mod.model import model_student_mod
from student_mod.dataset import *
import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

encoder_out_dim = 384


parser  = argparse.ArgumentParser()

parser.add_argument('--path_resume', type=str, default=None)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
# print(args.callbacks)

# print(args)

# assert False
print("-"*10)
print(args)
print("-"*10)

lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(monitor="train_loss", mode="min")

if args.path_resume:

    import wandb

    checkpoint_reference = args.path_resume

    # download checkpoint locally (if not already cached)
    run = wandb.init()
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()

    path_checkpoint = artifact_dir + '/model.ckpt'
    wandb.finish()


    wandb_logger = WandbLogger(project="Student_mod", name="mobilevit", log_model="all")
    
    # model = MocoModel()
    # path_checkpoint = "/content/epoch=38-step=34164.ckpt"
    model = BarlowTwins.load_from_checkpoint(path_checkpoint)
    trainer = Trainer.from_argparse_args(args, max_epochs=max_epochs,
                        accelerator="auto",
                        devices=1 if torch.cuda.is_available() else None,
                        #  default_root_dir="/content/drive/MyDrive/log_moco_sau",
                        resume_from_checkpoint=path_checkpoint,
                        #  limit_train_batches=20,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback, lr_monitor],
    )

else:
    # import wandb

    # checkpoint_reference = args.path_resume

    # # download checkpoint locally (if not already cached)
    # run = wandb.init()
    # artifact = run.use_artifact(checkpoint_reference, type="model")
    # artifact_dir = artifact.download()

    # path_checkpoint = artifact_dir + '/model.ckpt'
    # wandb.finish()


    wandb_logger = WandbLogger(project="Student_mod", name="mobilevit", log_model="all")
    # checkpoint_callback = ModelCheckpoint(monitor="train_loss_ssl", mode="min")
    model = model = BarlowTwins(
        # encoder=model_student_mod,
        encoder_out_dim=encoder_out_dim,
        num_training_samples=len(train_dataset),
        batch_size=batch_size,
        z_dim=z_dim,
    )
    # path_checkpoint = "/content/epoch=38-step=34164.ckpt"
    # model = MocoModel.load_from_checkpoint(path_checkpoint)
    trainer = Trainer.from_argparse_args(args, max_epochs=max_epochs,
                        accelerator="auto",
                        devices=1 if torch.cuda.is_available() else None,
                        #  default_root_dir="/content/drive/MyDrive/log_moco_sau",
                        #  resume_from_checkpoint=path_checkpoint,
                        #  limit_train_batches=20,
                        # precision=16,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback, lr_monitor],
    # , precision=16
    )


# # online_finetuner = OnlineFineTuner(encoder_output_dim=encoder_out_dim, num_classes=10)
# checkpoint_callback = ModelCheckpoint(every_n_epochs=10, save_top_k=-1, save_last=True)

# trainer = Trainer(
#     max_epochs=max_epochs,
#     accelerator="auto",
#     precision=16,
#     devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
#     callbacks=[
#                 # online_finetuner, 
#                 checkpoint_callback
#             ],
# )
trainer.fit(model, train_loader)