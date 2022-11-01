from student_mod.config import *
from cvnets.models.classification.mobilevit import MobileViT
import torch 
import torch.nn as nn
# from ghostpes.ghostnet import ghostnet, module_ghost_1, module_ghost_2




        

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# model_ghost = ghostnet()

model_student_mod = MobileViT(opts)
model_student_mod.fc = Identity()

# print(model_student_mod)

# model_ghost_git.layer_1 = nn.Sequential(
#         *list(model_ghost.blocks.children())[:-5],
#     )
# model_ghost_git.layer_2 = nn.Conv2d(40, 48, 1)
# model_ghost_git.layer_2 = Identity()
# model_ghost_git.layer_3[0] = Identity()
# model_ghost_git.layer_4[0] = module_ghost_1
# model_ghost_git.layer_5[0] = module_ghost_2


# print(model_ghost_git(torch.ones((2,3,224,224))).shape)