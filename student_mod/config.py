import argparse

num_workers = 2
img_size = 64
# batch_size = 192
# memory_bank_size = 2048
seed = 1
# max_epochs = 300

batch_size = 156
num_workers = 2  # to run notebook on CPU
max_epochs = 200
z_dim = 2048

data_dir_train = "data_celeba_processing"
data_dir_fine_x_train = "fdataX_train.npy"
data_dir_fine_y_train = "flabels_train.npy"
data_dir_fine_x_valid = "fdataX_valid.npy"
data_dir_fine_y_valid = "flabels_valid.npy"
data_dir_fine_x_test = "fdataX_test.npy"
data_dir_fine_y_test = "flabels_test.npy"
# path_to_train = "frame_video_sau"


opts = argparse.Namespace()

setattr(opts,"model.classification.name","mobilevit")
setattr(opts,"model.classification.classifier_dropout", 0.1)

setattr(opts,"model.classification.mit.mode" ,"x_small")
setattr(opts,"model.classification.mit.ffn_dropout", 0.0)
setattr(opts,"model.classification.mit.attn_dropout", 0.0)
setattr(opts,"model.classification.mit.dropout", 0.05)
setattr(opts,"model.classification.mit.number_heads", 4)
setattr(opts,"model.classification.mit.no_fuse_local_global_features", False)
setattr(opts,"model.classification.mit.conv_kernel_size", 3)

setattr(opts,"model.classification.activation.name", "swish")

setattr(opts,"model.normalization.name", "batch_norm_2d")
setattr(opts,"model.normalization.momentum", 0.1)

setattr(opts,"model.activation.name", "swish")

setattr(opts,"model.activation.layer.global_pool", "mean")
setattr(opts,"model.activation.layer.conv_init", "kaiming_normal")
setattr(opts,"model.activation.layer.linear_init", "trunc_normal")
setattr(opts,"model.activation.layer.linear_init_std_dev", 0.02)