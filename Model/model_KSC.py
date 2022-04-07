import torch.nn as nn
from Model.KSC_mul import encoder
import torch.nn.functional as F

class AttentionHSI(nn.Module):
   def __init__(self, num_classes, n_bands , dim):  #  num_classes为数据集中的类别数  n_bands为光谱段
        super(AttentionHSI, self).__init__()
        self.num_classes = num_classes
        self.n_bands = n_bands
        self.dim = dim
        self.Encoder= encoder()

        self.layer1_upfeatures = nn.Conv2d(dim, 2*dim, 1, padding=0, bias=True)

        self.layer4_downfeatures = nn.Conv2d(8*dim, 4*dim, 1, padding=0, bias=True)   # [1, 768, 5, 5] -> [1, 384, 5, 5]
        self.cls_pred = nn.Conv2d(2*dim, 14, 1)

        #decoder conv gn relu
        self.score1_donwnfeature = nn.Conv2d(4*dim, 2*dim, 1, padding=0, bias=True)


        self.score1_conv = nn.Conv2d(2*dim, 2*dim, 3 , 1, 1)
        self.score1_gn = nn.GroupNorm(6, 2*dim)
        self.score1_relu = nn.ReLU(inplace=True)

        self.score2_conv = nn.Conv2d(2*dim, 2*dim, 3 , 1, 1)
        self.score2_gn = nn.GroupNorm(6, 2*dim)
        self.score2_relu = nn.ReLU(inplace=True)

        self.score3_conv = nn.Conv2d(2*dim, 2*dim, 3 , 1, 1)
        self.score3_gn = nn.GroupNorm(6, 2*dim)
        self.score3_relu = nn.ReLU(inplace=True)

        self.score4_conv = nn.Conv2d(2*dim, 2*dim, 3 , 1, 1)
        self.score4_gn = nn.GroupNorm(6, 2*dim)
        self.score4_relu = nn.ReLU(inplace=True)

        self.layer1_up_conv = nn.Conv2d(dim, dim, 3 , 1, 1)
        self.layer1_up_gn = nn.GroupNorm(3, dim)
        self.layer1_up_relu = nn.ReLU(inplace=True)

        #输入
        self.input_conv = nn.Conv2d(176, 176, 3, 1, 1)
        self.input_gn = nn.GroupNorm(8, 176)
        self.input_relu = nn.ReLU(inplace=True)

        #修改shape
        self.score2_resize = nn.Conv2d(192, 192, (2, 2))
        self.score4_resize = nn.Conv2d(192, 192,(4, 2))
   def forward(self, x):  # x [1, 176, 181, 215]
       x  = self.input_conv(x)
       x  = self.input_gn(x)
       x  = self.input_relu(x)
       outs_list = self.Encoder(x)

       # decoder
       score1 = self.layer4_downfeatures(outs_list[3])  #    [1,768, 6, 7] -> [1, 384, 6, 7]
       score1_upsample = F.interpolate(score1, scale_factor=2.0, mode='bilinear', recompute_scale_factor=True, align_corners=True)  # [1, 384, 12, 14]
       score1 = outs_list[2] + score1_upsample  # [1, 384, 12, 14]
       score1 = self.score1_donwnfeature(score1)  # [1, 192, 12, 14]
       score1 = self.score1_conv(score1)
       score1 = self.score1_gn(score1)
       score1 = self.score1_relu(score1)

       score2 = F.interpolate(score1, scale_factor=(2), mode='bilinear', recompute_scale_factor=True, align_corners=True) #[1, 192, 24, 28]
       score2 = self.score2_resize(score2)  # [1, 192, 24, 28] -> [1, 192, 23 ,27]
       score2 = score2 + outs_list[1]  # [1, 192, 23, 27]
       score2 = self.score2_conv(score2)
       score2 = self.score2_gn(score2)
       score2 = self.score2_relu(score2) # [1, 192, 23 ,27]


       score3 = F.interpolate(score2, scale_factor=(2), mode='bilinear', recompute_scale_factor=True, align_corners=True) # [1, 192, 46, 54]
       layer1 = self.layer1_upfeatures(outs_list[0]) # [1,92, 46, 54] -> [1, 192, 46, 54]
       score3 = score3+layer1 #[1,192, 46, 54]
       score3 = self.score3_conv(score3)
       score3 = self.score3_gn(score3)
       score3 = self.score3_relu(score3)  #[1, 192, 46, 54]


       score4 = F.interpolate(score3, scale_factor=(4), mode='bilinear', recompute_scale_factor=True, align_corners=True)  # [1, 192, 184, 216]
       score4 = self.score4_resize(score4)  # [1, 192, 184, 216] -> [1, 192, 181, 215]
       score4 = self.score4_conv(score4)
       score4 = self.score4_gn(score4)
       score4 = self.score4_relu(score4)  # [1, 192, 181, 215]



       out = self.cls_pred(score4) #[1, 14, 181, 215]
       return out







