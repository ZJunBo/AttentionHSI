import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.IP_cat import SwinTransformer
torch.set_printoptions(profile="full")

class SwinHSI(nn.Module):
   def __init__(self, num_classes, n_bands , dim):  #  num_classes为数据集中的类别数  n_bands为光谱段
        super(SwinHSI, self).__init__()
        self.num_classes = num_classes
        self.n_bands = n_bands
        self.dim = dim
        self.swin = SwinTransformer()

        #输入
        self.input_conv = nn.Conv2d(200, 200, 3, 1, 1)
        self.input_gn = nn.GroupNorm(8, 200)
        self.input_relu = nn.ReLU(inplace=True)

        #
        self.layer1_upfeatures = nn.Conv2d(dim, 4*dim, 1, padding=0, bias=True)
        self.layer2_upfeatures = nn.Conv2d(2*dim, 4*dim, 1, padding=0, bias=True)
        self.layer4_downfeatures = nn.Conv2d(8*dim, 4*dim, 1, padding=0, bias=True)
        self.cls_pred = nn.Conv2d(680, 17, 1)

        #decoder conv gn relu
        self.score1_conv = nn.Conv2d(4*dim, 4*dim, 3 , 1, 1)
        self.score1_gn = nn.GroupNorm(12, 4*dim)
        self.score1_relu = nn.ReLU(inplace=True)

        self.score2_conv = nn.Conv2d(4*dim, 4*dim, 3 , 1, 1)
        self.score2_gn = nn.GroupNorm(12, 4*dim)
        self.score2_relu = nn.ReLU(inplace=True)

        self.score3_conv = nn.Conv2d(4*dim, 4*dim, 3 , 1, 1)
        self.score3_gn = nn.GroupNorm(12, 4*dim)
        self.score3_relu = nn.ReLU(inplace=True)

        self.score4_conv = nn.Conv2d(4*dim, 4*dim, 3 , 1, 1)
        self.score4_gn = nn.GroupNorm(12, 4*dim)
        self.score4_relu = nn.ReLU(inplace=True)

        self.layer1_up_conv = nn.Conv2d(dim, dim, 3 , 1, 1)
        self.layer1_up_gn = nn.GroupNorm(3, dim)
        self.layer1_up_relu = nn.ReLU(inplace=True)




   def forward(self, x):
       x  = self.input_conv(x)
       x  = self.input_gn(x)
       x  = self.input_relu(x) # [1, 200, 145, 145]
       outs_list = self.swin(x)

       # decoder
       score1 = self.layer4_downfeatures(outs_list[3])  # [1, 384 , 5, 5]
       score1_upsample = F.interpolate(score1, scale_factor=2.0, mode='bilinear', recompute_scale_factor=True, align_corners=True)  # [1, 384, 10, 10]
       score1 = outs_list[2] + score1_upsample  # [1, 384, 10, 10]

       score1 = self.score1_conv(score1)
       score1 = self.score1_gn(score1)
       score1 = self.score1_relu(score1) #  [1, 384, 10, 10]

       score2 = F.interpolate(score1, scale_factor=(19/10), mode='bilinear', recompute_scale_factor=True, align_corners=True) #[1, 384, 19, 19]
       layer2 = self.layer2_upfeatures(outs_list[1]) # [1, 192, 19, 19] -> [1, 384, 19, 19]
       score2 = score2 +layer2

       score2 = self.score2_conv(score2)
       score2 = self.score2_gn(score2)
       score2 = self.score2_relu(score2) # [1, 384, 19 ,19]

       score3 = F.interpolate(score2, scale_factor=(37/19), mode='bilinear', recompute_scale_factor=True, align_corners=True) # [1, 384, 37, 37]
       layer1 = self.layer1_upfeatures(outs_list[0]) # [1,96,37,37] -> [1,384,37,37]
       score3 = score3+layer1 #[1,384, 37, 37]

       score3 = self.score3_conv(score3)
       score3 = self.score3_gn(score3)
       score3 = self.score3_relu(score3)

       score4 = F.interpolate(score3, scale_factor=(145/37), mode='bilinear', recompute_scale_factor=True, align_corners=True)  # [1, 384, 145, 145]
       score4 = self.score4_conv(score4)
       score4 = self.score4_gn(score4)
       score4 = self.score4_relu(score4)

       layer1_up = F.interpolate(outs_list[0], scale_factor=(145/37), mode='bilinear', recompute_scale_factor=True, align_corners=True) #[1, 96, 145, 145]
       layer1_up = self.layer1_up_conv(layer1_up)
       layer1_up = self.layer1_up_gn(layer1_up)
       layer1_up = self.layer1_up_relu(layer1_up)

       INF = 1
       out = torch.cat([score4, layer1_up*INF], dim=1)
       out = torch.cat([out, x], 1)

       out = self.cls_pred(out) #[1, 17, 145, 145]
       return out







