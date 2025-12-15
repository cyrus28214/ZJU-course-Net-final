#!/usr/bin/env python
# encoding: utf-8
"""
Neural Network Models for Semantic Communication System
包含分类器、编码器-解码器和端到端系统模型
"""

import torch
import torch.nn.functional as F


class MLP_MNIST(torch.nn.Module):
    """MNIST手写数字分类器
    
    4层全连接神经网络，用于分类重建后的图像
    """
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 500)
        self.fc2 = torch.nn.Linear(500, 250)
        self.fc3 = torch.nn.Linear(250, 125)
        self.fc4 = torch.nn.Linear(125, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MLP_Encoder_Decoder(torch.nn.Module):
    """语义通信编码器-解码器
    
    将原始图像压缩为低维语义特征，然后重建
    
    Args:
        channel: 语义特征维度 (默认78，对应压缩率0.1)
    """
    def __init__(self, channel=78):
        super(MLP_Encoder_Decoder, self).__init__()
        # 编码器: 784 -> 1024 -> channel
        self.fc1_1 = torch.nn.Linear(28 * 28, 1024)
        self.fc1_2 = torch.nn.Linear(1024, channel)
        # 解码器: channel -> 1024 -> 784
        self.fc2_1 = torch.nn.Linear(channel, 1024)
        self.fc2_2 = torch.nn.Linear(1024, 28 * 28)
    
    def encode(self, x):
        """编码: 提取语义特征"""
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc1_2(x))
        return x
    
    def decode(self, x):
        """解码: 从语义特征重建图像"""
        x = F.relu(self.fc2_1(x))
        x = F.relu(self.fc2_2(x))
        return x
    
    def forward(self, x):
        """完整的编码-解码流程"""
        x = self.encode(x)
        x = self.decode(x)
        return x


class SemanticCommSystem(torch.nn.Module):
    """端到端语义通信系统
    
    组合编码器-解码器和分类器，用于端到端攻击
    """
    def __init__(self, encoder_decoder, classifier):
        super().__init__()
        self.coder = encoder_decoder
        self.classifier = classifier
    
    def forward(self, x):
        semantic = self.coder.encode(x)
        reconstructed = self.coder.decode(semantic)
        outputs = self.classifier(reconstructed)
        return outputs
