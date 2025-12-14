#!/usr/bin/env python
# encoding: utf-8
"""
语义空间攻击
在语义特征空间中进行攻击，而不是直接在图像空间
"""

import torch
import torch.nn as nn


def end_to_end_attack(encoder, decoder, classifier, images, labels, 
                      epsilon=0.3, alpha=0.01, num_iter=40,
                      lambda_recon=0.5, lambda_class=1.0):
    """
    端到端对抗攻击
    优化目标：误导分类 + 保持图像相似度
    
    参数:
        encoder: 编码器
        decoder: 解码器
        classifier: 分类器
        images: 输入图像
        labels: 真实标签
        epsilon: 最大扰动
        alpha: 步长
        num_iter: 迭代次数
        lambda_recon: 重建损失权重
        lambda_class: 分类损失权重
    
    返回:
        adv_images: 对抗样本
        perturbation: 扰动
        metrics: 攻击指标字典
    """
    device = images.device
    original_images = images.clone().detach()
    adv_images = images.clone().detach()
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    
    attack_success = []
    
    for i in range(num_iter):
        adv_images = adv_images.clone().detach().requires_grad_(True)
        
        # 完整流程
        semantic_features = encoder(adv_images)
        reconstructed = decoder(semantic_features)
        outputs = classifier(reconstructed)
        
        # 计算损失
        # 1. 分类损失（最大化）
        loss_class = criterion_ce(outputs, labels)
        
        # 2. 重建损失（最小化，保持图像相似）
        loss_recon = criterion_mse(adv_images, original_images)
        
        # 组合损失
        loss = lambda_class * loss_class - lambda_recon * loss_recon
        
        # 反向传播
        if hasattr(encoder, 'zero_grad'):
            encoder.zero_grad()
        if hasattr(decoder, 'zero_grad'):
            decoder.zero_grad()
        if hasattr(classifier, 'zero_grad'):
            classifier.zero_grad()
        loss.backward()
        
        # 更新
        grad = adv_images.grad
        adv_images = adv_images + alpha * grad.sign()
        
        # 投影
        perturbation = torch.clamp(adv_images - original_images, -epsilon, epsilon)
        adv_images = original_images + perturbation
        adv_images = torch.clamp(adv_images, 0, 1)
        
        # 记录攻击成功率
        with torch.no_grad():
            sf = encoder(adv_images)
            rec = decoder(sf)
            out = classifier(rec)
            pred = out.argmax(dim=1)
            success = (pred != labels).float().mean().item()
            attack_success.append(success)
    
    perturbation = adv_images - original_images
    
    metrics = {
        'attack_success_rate': attack_success[-1],
        'attack_success_history': attack_success,
        'perturbation_l2': torch.norm(perturbation, p=2).item(),
        'perturbation_linf': torch.norm(perturbation, p=float('inf')).item()
    }
    
    return adv_images.detach(), perturbation.detach(), metrics


def semantic_feature_attack(encoder, decoder, classifier, images, labels,
                            epsilon_semantic=1.0, num_iter=20):
    """
    语义特征空间攻击
    直接在语义特征上添加扰动
    
    参数:
        encoder: 编码器
        decoder: 解码器
        classifier: 分类器
        images: 输入图像
        labels: 真实标签
        epsilon_semantic: 语义特征扰动大小
        num_iter: 迭代次数
    
    返回:
        adv_images: 对抗样本（通过扰动语义特征后解码得到）
        semantic_perturbation: 语义特征扰动
    """
    device = images.device
    
    # 提取原始语义特征
    with torch.no_grad():
        original_semantic = encoder(images)
    
    # 初始化扰动
    semantic_perturbation = torch.zeros_like(original_semantic).requires_grad_(True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([semantic_perturbation], lr=0.01)
    
    for i in range(num_iter):
        # 扰动后的语义特征
        perturbed_semantic = original_semantic + semantic_perturbation
        
        # 解码和分类
        reconstructed = decoder(perturbed_semantic)
        outputs = classifier(reconstructed)
        
        # 损失
        loss = criterion(outputs, labels)
        
        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 限制扰动大小
        with torch.no_grad():
            semantic_perturbation.data = torch.clamp(semantic_perturbation.data, 
                                                    -epsilon_semantic, epsilon_semantic)
    
    # 生成最终对抗样本
    with torch.no_grad():
        final_semantic = original_semantic + semantic_perturbation
        adv_images = decoder(final_semantic)
        adv_images = torch.clamp(adv_images, 0, 1)
    
    return adv_images, semantic_perturbation.detach()


def channel_noise_attack(encoder, decoder, classifier, images, labels,
                        noise_power=0.1, num_samples=10):
    """
    信道噪声攻击
    模拟攻击者在信道中注入对抗噪声
    
    参数:
        encoder: 编码器
        decoder: 解码器
        classifier: 分类器
        images: 输入图像
        labels: 真实标签
        noise_power: 噪声功率
        num_samples: 采样次数
    
    返回:
        best_adv_images: 最佳对抗样本
        best_noise: 最佳噪声
    """
    device = images.device
    
    # 编码
    with torch.no_grad():
        semantic_features = encoder(images)
    
    best_success_rate = 0
    best_adv_images = images.clone()
    best_noise = None
    
    criterion = nn.CrossEntropyLoss()
    
    for sample_idx in range(num_samples):
        # 生成随机噪声
        noise = torch.randn_like(semantic_features) * noise_power
        noise.requires_grad = True
        
        # 优化噪声
        optimizer = torch.optim.Adam([noise], lr=0.01)
        
        for step in range(10):
            noisy_features = semantic_features + noise
            reconstructed = decoder(noisy_features)
            outputs = classifier(reconstructed)
            
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 限制噪声功率
            with torch.no_grad():
                noise_norm = torch.norm(noise)
                if noise_norm > noise_power * semantic_features.numel() ** 0.5:
                    noise.data = noise.data / noise_norm * noise_power * semantic_features.numel() ** 0.5
        
        # 评估
        with torch.no_grad():
            final_features = semantic_features + noise
            adv_images = decoder(final_features)
            outputs = classifier(adv_images)
            pred = outputs.argmax(dim=1)
            success_rate = (pred != labels).float().mean().item()
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_adv_images = adv_images.clone()
                best_noise = noise.clone()
    
    return best_adv_images, best_noise

