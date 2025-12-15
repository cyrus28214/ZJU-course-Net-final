#!/usr/bin/env python
# encoding: utf-8
"""
FGSM (Fast Gradient Sign Method) 对抗攻击
针对语义通信系统的快速对抗攻击实现
"""

import torch
import torch.nn as nn

def fgsm_attack(model, images, labels, epsilon=0.3, targeted=False, target_labels=None):
    """
    FGSM对抗攻击
    
    参数:
        model: 目标模型（分类器或端到端模型）
        images: 输入图像 (batch_size, channels, height, width)
        labels: 真实标签 (batch_size,)
        epsilon: 扰动大小
        targeted: 是否为目标攻击
        target_labels: 目标标签（仅在targeted=True时使用）
    
    返回:
        adv_images: 对抗样本
        perturbation: 添加的扰动
    """
    # 确保图像需要梯度
    images = images.clone().detach()
    images.requires_grad = True
    
    # 前向传播
    outputs = model(images)
    
    # 计算损失
    criterion = nn.CrossEntropyLoss()
    if targeted:
        # 目标攻击：最小化到目标类别的损失
        assert target_labels is not None, "目标攻击需要提供target_labels"
        loss = -criterion(outputs, target_labels)
    else:
        # 非目标攻击：最大化原始类别的损失
        loss = criterion(outputs, labels)
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 生成对抗扰动
    grad_sign = images.grad.sign()
    perturbation = epsilon * grad_sign
    
    # 生成对抗样本
    adv_images = images + perturbation
    adv_images = torch.clamp(adv_images, 0, 1)  # 确保像素值在[0,1]范围内
    
    return adv_images.detach(), perturbation.detach()


def fgsm_attack_semantic(encoder, decoder, classifier, images, labels, epsilon=0.3):
    """
    针对语义通信系统的FGSM攻击
    攻击整个编码-解码-分类流程
    
    参数:
        encoder: 语义编码器
        decoder: 语义解码器
        classifier: 分类器
        images: 输入图像
        labels: 真实标签
        epsilon: 扰动大小
    
    返回:
        adv_images: 对抗样本
        perturbation: 添加的扰动
        semantic_features: 对抗样本的语义特征
    """
    device = images.device
    images = images.clone().detach()
    images.requires_grad = True
    
    # 完整的语义通信流程
    # 编码
    semantic_features = encoder(images)
    
    # 解码
    reconstructed = decoder(semantic_features)
    
    # 分类
    outputs = classifier(reconstructed)
    
    # 计算损失
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    
    # 反向传播
    encoder.zero_grad()
    decoder.zero_grad()
    classifier.zero_grad()
    loss.backward()
    
    # 生成对抗扰动
    grad_sign = images.grad.sign()
    perturbation = epsilon * grad_sign
    
    # 生成对抗样本
    adv_images = images + perturbation
    adv_images = torch.clamp(adv_images, 0, 1)
    
    # 获取对抗样本的语义特征
    with torch.no_grad():
        adv_semantic_features = encoder(adv_images)
    
    return adv_images.detach(), perturbation.detach(), adv_semantic_features.detach()

