"""
模型定义
"""
import torch
from torch import nn
import torchvision.models as models
import yaml



class GridAnchorDetector(nn.Module):
    #输入shape:[B,3,416,416]
    #输出shape:[B,13,13,5,(5+nc)],其中(5+nc)包括(tx,ty,tw,th,conf,nc个onehot)
    def __init__(self):
        super(GridAnchorDetector,self).__init__()
        resnet=models.resnet18()
        with open("models/config.yaml",'r') as f:
            config=yaml.safe_load(f)
        self.nc=config["nc"]#类别数
        #1、骨干网络(用于提取特征)
        self.backbone=nn.Sequential(*list(resnet.children())[:-2])#除去resnet最后两层
        #shape:[B,512,13,13]

        #2、特征增强(让特征复杂一些)
        self.enhance=nn.Sequential(
            nn.Conv2d(512,1024,3,1,1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            #shape:[B,1024,13,13]
            nn.Conv2d(1024,512,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            #shape:[B,512,13,13]
            nn.Conv2d(512,1024,3,1,1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            #shape:[B,1024,13,13]
        )

        #3、特征处理(将特征转换为最终格式)
        self.process=nn.Sequential(
            nn.Conv2d(1024,5*(5+self.nc),1)
            #shape:[B,5*(5+nc),13,13]
        )

    def forward(self,x):
        x=self.backbone(x)
        x=self.enhance(x)
        x=self.process(x)
        x=x.permute(0,2,3,1)#重排为[B,13,13,5*(5+nc)]
        B,grid_y_size,grid_x_size,_=x.shape
        x=x.view(B,grid_y_size,grid_x_size,5,5+self.nc)#重塑为[B,13,13,5,(5+nc)]
        x[...,0]=torch.sigmoid(x[...,0])#tx用sigmoid归一,得到σ(tx)
        x[...,1]=torch.sigmoid(x[...,1])#ty用sigmoid归一,得到σ(ty)
        x[...,4]=torch.sigmoid(x[...,4])#置信度conf用sigmoid归一
        #x[...,5:5+self.nc]=torch.softmax(x[...,5:5+self.nc],-1)#分类用softmax,CrossEntropyLoss损失涵盖此过程
        return x