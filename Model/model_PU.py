import torch
import torch.nn as nn
import numpy as np
from Model.PU_mul import encoder
import torch.nn.functional as F
np.set_printoptions(threshold=np.inf)

class AttentionHSI(nn.Module):
   def __init__(self, num_classes, n_bands , dim):  #  num_classes为数据集中的类别数  n_bands为光谱段
        super(AttentionHSI, self).__init__()
        self.num_classes = num_classes
        self.n_bands = n_bands
        self.dim = dim
        self.Encoder = encoder()
        self.layer4_downfeatures = nn.Conv2d(8*dim, 4*dim, 1, padding=0, bias=True)   # [1, 768, 5, 5] -> [1, 384, 5, 5]
        self.cls_pred = nn.Conv2d(3*dim, 10, 1)


        #decoder conv gn relu
        self.score1_conv = nn.Conv2d(4*dim, 4*dim, 3 , 1, 1)
        self.score1_gn = nn.GroupNorm(12, 4*dim)
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
        self.input_conv = nn.Conv2d(103, 96, 3, 1, 1)
        self.input_gn = nn.GroupNorm(3, 96)
        self.input_relu = nn.ReLU(inplace=True)

        # 通过卷积修改shape
        self.score1_downfeature = nn.Conv2d(4 * dim, 2 * dim, 1, padding=0, bias=True)
        self.score2_resize = nn.Conv2d(2*dim, 2*dim, (1, 2))
        self.score3_resize = nn.Conv2d(2*dim, 2*dim, (2, 2))
        self.score3_downfeature = nn.Conv2d(3*dim, 2*dim, 1 ,padding=0, bias=True)
        self.score4_downfeature = nn.Conv2d(2*dim, 1*dim, 1, padding=0, bias=True)
        self.out_resize = nn.Conv2d(2*dim, 2*dim, (2, 1))
        self.layer1_upfeature = nn.Conv2d(dim, 2*dim, 1, padding=0, bias=True)

   def forward(self, x):
       x  = self.input_conv(x)
       x  = self.input_gn(x)
       x  = self.input_relu(x)
       outs_list = self.Encoder(x)

       # decoder
       score1 = self.layer4_downfeatures(outs_list[3])  # [1, 384 , 10, 6]
       score1_upsample = F.interpolate(score1, scale_factor=2.0, mode='bilinear', recompute_scale_factor=True, align_corners=True)  # [1, 384, 20, 12]
       score1 = outs_list[2] + score1_upsample  # [1, 384, 20, 12] 信息融合

       score1 = self.score1_conv(score1)
       score1 = self.score1_gn(score1)
       score1 = self.score1_relu(score1) #   [1, 384, 20, 12]
       score1 = self.score1_downfeature(score1)  # [1, 192, 20, 12]

       score2 = F.interpolate(score1, scale_factor=2.0, mode='bilinear', recompute_scale_factor=True, align_corners=True) #[1, 192, 40, 24]
       score2 = self.score2_resize(score2) # [1, 192 ,40, 23]
       score2 = score2 + outs_list[1]

       score2 = self.score2_conv(score2)
       score2 = self.score2_gn(score2)
       score2 = self.score2_relu(score2) # [1, 192, 40 ,23]


       score3 = F.interpolate(score2, scale_factor=2.0, mode='bilinear', recompute_scale_factor=True, align_corners=True) # [1, 192, 80, 46]
       score3 = self.score3_resize(score3)  # [1, 192, 79 ,45]
       score3 = torch.cat([score3, outs_list[0]], 1)  # [1, 288, 79, 45]
       score3 = self.score3_downfeature(score3) # [1, 192, 79, 45]
       score3 = self.score3_conv(score3)
       score3 = self.score3_gn(score3)
       score3 = self.score3_relu(score3)


       score4 = F.interpolate(score3, scale_factor=4.0, mode='bilinear', recompute_scale_factor=True, align_corners=True)  # [1, 192, 316, 180]
       score4 = self.score4_conv(score4)
       score4 = self.score4_gn(score4)
       score4 = self.score4_relu(score4)
       score4 = self.out_resize(score4)  # [1, 192 ,315 ,180]

       out = torch.cat([score4, x*1.5], 1)
       out = self.cls_pred(out) #[1, 10, 315, 180]
       return out




