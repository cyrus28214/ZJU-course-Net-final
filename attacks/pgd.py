#!/usr/bin/env python
# encoding: utf-8
"""
PGD (Projected Gradient Descent) 对抗攻击
迭代式对抗攻击，比FGSM更强大
"""

import torch
import torch.nn as nn


def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, num_iter=40, 
               targeted=False, target_labels=None, random_start=True):
    """
    PGD对抗攻击
    
    参数:
        model: 目标模型
        images: 输入图像
        labels: 真实标签
        epsilon: 最大扰动幅度（L∞范数）
        alpha: 每步的步长
        num_iter: 迭代次数
        targeted: 是否为目标攻击
        target_labels: 目标标签
        random_start: 是否随机初始化
    
    返回:
        adv_images: 对抗样本
        perturbation: 总扰动
    """
    device = images.device
    original_images = images.clone().detach()
    
    # 随机初始化
    if random_start:
        adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, 0, 1)
    else:
        adv_images = images.clone()
    
    criterion = nn.CrossEntropyLoss()
    
    for i in range(num_iter):
        adv_images = adv_images.clone().detach().requires_grad_(True)
        
        # 前向传播
        outputs = model(adv_images)
        
        # 计算损失
        if targeted:
            assert target_labels is not None
            loss = -criterion(outputs, target_labels)
        else:
            loss = criterion(outputs, labels)
        
        # 反向传播
        model.zero_grad()
        loss.backward()
        
        # 更新对抗样本
        grad = adv_images.grad
        adv_images = adv_images + alpha * grad.sign()
        
        # 投影到epsilon球内
        perturbation = torch.clamp(adv_images - original_images, -epsilon, epsilon)
        adv_images = original_images + perturbation
        
        # 确保在有效范围内
        adv_images = torch.clamp(adv_images, 0, 1)
    
    perturbation = adv_images - original_images
    return adv_images.detach(), perturbation.detach()


def pgd_attack_semantic(encoder, decoder, classifier, images, labels, 
                        epsilon=0.3, alpha=0.01, num_iter=40, random_start=True):
    """
    针对语义通信系统的PGD攻击
    
    参数:
        encoder: 语义编码器
        decoder: 语义解码器  
        classifier: 分类器
        images: 输入图像
        labels: 真实标签
        epsilon: 最大扰动幅度
        alpha: 步长
        num_iter: 迭代次数
        random_start: 随机初始化
    
    返回:
        adv_images: 对抗样本
        perturbation: 扰动
    """
    device = images.device
    original_images = images.clone().detach()
    
    # 随机初始化
    if random_start:
        adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, 0, 1)
    else:
        adv_images = images.clone()
    
    criterion = nn.CrossEntropyLoss()
    
    for i in range(num_iter):
        adv_images = adv_images.clone().detach().requires_grad_(True)
        
        # 完整的语义通信流程
        semantic_features = encoder(adv_images)
        reconstructed = decoder(semantic_features)
        outputs = classifier(reconstructed)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        if hasattr(encoder, 'zero_grad'):
            encoder.zero_grad()
        if hasattr(decoder, 'zero_grad'):
            decoder.zero_grad()
        if hasattr(classifier, 'zero_grad'):
            classifier.zero_grad()
        loss.backward()
        
        # 更新对抗样本
        grad = adv_images.grad
        adv_images = adv_images + alpha * grad.sign()
        
        # 投影到epsilon球内
        perturbation = torch.clamp(adv_images - original_images, -epsilon, epsilon)
        adv_images = original_images + perturbation
        
        # 确保在有效范围内
        adv_images = torch.clamp(adv_images, 0, 1)
    
    perturbation = adv_images - original_images
    return adv_images.detach(), perturbation.detach()

