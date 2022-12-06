from student_mod.config import *
from cvnets.models.classification.mobilevit import MobileViT
import torch 
import torch.nn as nn
from cvnets.modules import InvertedResidual
from cvnets.layers import ConvLayer
# from ghostpes.ghostnet import ghostnet, module_ghost_1, module_ghost_2




        

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# model_ghost = ghostnet()
def get_mobile_vit(pretrained=False):
    model = MobileViT(opts)
    if pretrained:
        model.load_state_dict(torch.load("mobilevit_xs.pt"))
        
    model.classifier.fc = Identity()
    model.conv_1 = ConvLayer(
                opts=opts,
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                use_norm=True,
                use_act=True,
            )
    #  = ConvLayer(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, normalization=BatchNorm2d, activation=Swish, bias=False)

    model.layer_2[0] = InvertedResidual(opts, in_channels=32, out_channels=48, stride=1, expand_ratio=4, dilation=1, skip_conn=False)

    
    return model

# print(model_student_mod(torch.ones((2,3,224,224))))

# model_ghost_git.layer_1 = nn.Sequential(
#         *list(model_ghost.blocks.children())[:-5],
#     )
# model_ghost_git.layer_2 = nn.Conv2d(40, 48, 1)
# model_ghost_git.layer_2 = Identity()
# model_ghost_git.layer_3[0] = Identity()
# model_ghost_git.layer_4[0] = module_ghost_1
# model_ghost_git.layer_5[0] = module_ghost_2


# print(model_ghost_git(torch.ones((2,3,224,224))).shape)

print(model_student_mod)