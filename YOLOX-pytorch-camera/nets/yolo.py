#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv ,Inception_BaseConv
from nets.attention import cbam_block, eca_block, se_block, CoordAtt
attention_block = [se_block, cbam_block, eca_block, CoordAtt]

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [256, 512, 1024], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
#------------------INCEPTION 结构---------------------
        Conv_I = Inception_BaseConv
#——————————————————————————————————————————————
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
    #-------------------inception结构------------------
            #self.stems.append(Inception_BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            #源码
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
    #——————————————————————————————————————————————————————————————————————————————————————————
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu",attention=0):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features
        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")
#——————————————————————————————加入注意力————————————————————————————————————————————
        if attention >= 5:
            raise AssertionError("zyl must be less than or equal to 3 (0, 1, 2, 3).")
        #----------添加注意力机制------------
        self.attention     =  attention
        #attention_block = [se_block, cbam_block, eca_block，坐标注意力]
        if 1 <= self.attention and self.attention <= 4:
            #  nano 模型的宽度（通道数）为L模型（256,512,1024）的1/4，（64,128,256）
            #  tiny 模型的宽度（通道数）为L模型（256,512,1024）的0.375，（96,192,384）
            #   s   模型的宽度（通道数）为L模型（256,512,1024）的1/2,（128,256,512）
            self.feat1_att      = attention_block[self.attention - 1](128)
            self.feat2_att      = attention_block[self.attention - 1](256)
            self.feat3_att      = attention_block[self.attention - 1](512)
#————————————————————————————————————原模型——————————————————————————————————————————————
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        # -------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        # -------------------------------------------#
        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)

#————————————————————————————————————Bifpn——————————————————————————————————
        # # p3_out进行1*1卷积：80,256----80，512
        # self.lateral_conv0 = BaseConv(int(in_channels[0] * width), int(in_channels[1] * width), 1, 1, act=act)
        # # p3和p4融合后进行1*1卷积：40,40,512,----40,40,1024
        # self.lateral_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[2] * width), 1, 1, act=act)
        # # 自下而上过程 p5_out与p4_zhong结合  20,1024 ---20,512
        # self.lateral_conv2 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        # # --其中self.bu_conv2 需要修改下面
        # self.bu_conv2 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        # self.bu_conv3 = Conv(int(in_channels[2] * width), int(in_channels[2] * width), 3, 2, act=act)
        # #  2048--->1024
        # self.p5_out = CSPLayer(
        #     int(2 * in_channels[2] * width),
        #     int(in_channels[2] * width),
        #     round(3 * depth),
        #     False,
        #     depthwise=depthwise,
        #     act=act,
        # )
#——————————————————————————————————————————————————————————————————————————————————————————————
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        # #-------------------------------------------#
        # #   80, 80, 256 -> 40, 40, 256
        # #-------------------------------------------#
        # self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        # #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]
#————————————————————注意力——————————————————————————
        if 1 <= self.attention and self.attention <= 4:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)
            feat3 = self.feat3_att(feat3)
#-----------------------------------------------------
#——————————————————————————————————————————————————————————————————
#-----------------------Bifpn--------------------------------------
        # #   80,256--->80,512
        # p3 = self.lateral_conv0(feat1)
        # #   80,512--->40,512(下采样）
        # p3_p4 = self.bu_conv2(p3)
        # #   40,512 + 40,512 --->40,512
        # p4_zhong = torch.cat([p3_p4, feat2], 1)
        # #   40,1024---40,512
        # p4_zhong = self.C3_p4(p4_zhong)
        # #   40,512 --->40,1024
        # p4_zhong_conv = self.lateral_conv1(p4_zhong)
        # #   40,1024 --->20,1024
        # p4_p5 = self.bu_conv3(p4_zhong_conv)
        # #   20,1024 + 20,1024 --->20,1024
        # p5_zhong = torch.cat([p4_p5, feat3], 1)
        #
        # P5_out = self.p5_out(p5_zhong)
        #
        # #   20,1024 --->20,512
        # p5_p4 = self.lateral_conv2(P5_out)
        # #   20,512--->40,512
        # p5_p4_shang = self.upsample(p5_p4)
        # #   40,512  (((((可以做文章）））））
        # #   40,512 + 40,512 ---->40,512
        # p4_zhong_feat1 = torch.cat([p4_zhong, feat2], 1)
        # #   40,1024---40,512
        # p4_zhong_feat1 = self.C3_p4(p4_zhong_feat1)
        #
        # P4_out = p5_p4_shang + p4_zhong_feat1
        #
        # #   40,512--->40,256
        # p4_c1 = self.reduce_conv1(P4_out)
        # #   40,256--->80,256
        # p4_p3_shang = self.upsample(p4_c1)
        # #   80,256 + 80,256 --->80,256
        # p3_zhong = torch.cat([p4_p3_shang, feat1], 1)
        #
        # P3_out = self.C3_p3(p3_zhong)
#----------------------------END---------------------------

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)  

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        P3_downsample   = self.bu_conv2(P3_out) 
        #-------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P3_downsample   = torch.cat([P3_downsample, P4], 1) 
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P4_out          = self.C3_n3(P3_downsample) 

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        P4_downsample   = self.bu_conv1(P4_out)
        #-------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        #-------------------------------------------#
        P4_downsample   = torch.cat([P4_downsample, P5], 1)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        P5_out          = self.C3_n4(P4_downsample)

        return (P3_out, P4_out, P5_out)

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False 
#——————————————————————————————————————————————————————————————————
        # ----------------------添加的注意力-----------------------------
        #   attention = 0--->不使用注意力机制
        #   attention = 1--->[se_block]
        #   attention = 2--->[cbam_block]
        #   attention = 3--->[eca_block]
        #   attention = 4--->[CoordAtt]
#——————————————————————————————————————————————————————————————————
        self.backbone   = YOLOPAFPN(depth, width, depthwise=depthwise,attention=0)
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs    = self.backbone.forward(x)
        outputs     = self.head.forward(fpn_outs)
        return outputs
