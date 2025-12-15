#!/usr/bin/env python
# encoding: utf-8
"""
攻击评估工具
评估对抗攻击的效果和质量
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def evaluate_attack(model, clean_images, adv_images, labels, device='cuda'):
    """
    评估对抗攻击效果
    
    参数:
        model: 目标模型
        clean_images: 干净图像
        adv_images: 对抗样本
        labels: 真实标签
        device: 设备
    
    返回:
        metrics: 评估指标字典
    """
    model.eval()
    
    with torch.no_grad():
        # 干净图像的预测
        clean_outputs = model(clean_images)
        clean_pred = clean_outputs.argmax(dim=1)
        clean_acc = (clean_pred == labels).float().mean().item()
        
        # 对抗样本的预测
        adv_outputs = model(adv_images)
        adv_pred = adv_outputs.argmax(dim=1)
        adv_acc = (adv_pred == labels).float().mean().item()
        
        # 攻击成功率
        attack_success_rate = (adv_pred != labels).float().mean().item()
    
    # 计算扰动统计
    perturbation = (adv_images - clean_images).cpu().numpy()
    
    metrics = {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'attack_success_rate': attack_success_rate,
        'perturbation_l2_mean': np.mean(np.linalg.norm(perturbation.reshape(perturbation.shape[0], -1), ord=2, axis=1)),
        'perturbation_l2_max': np.max(np.linalg.norm(perturbation.reshape(perturbation.shape[0], -1), ord=2, axis=1)),
        'perturbation_linf_mean': np.mean(np.abs(perturbation)),
        'perturbation_linf_max': np.max(np.abs(perturbation)),
    }
    
    return metrics


def evaluate_semantic_attack(encoder, decoder, classifier, clean_images, adv_images, labels, device='cuda'):
    """
    评估针对语义通信系统的攻击
    
    参数:
        encoder: 编码器（模型或函数）
        decoder: 解码器（模型或函数）
        classifier: 分类器
        clean_images: 干净图像
        adv_images: 对抗样本
        labels: 真实标签
        device: 设备
    
    返回:
        metrics: 评估指标字典
    """
    # 如果是模型对象，设置为eval模式
    if hasattr(encoder, 'eval'):
        encoder.eval()
    if hasattr(decoder, 'eval'):
        decoder.eval()
    if hasattr(classifier, 'eval'):
        classifier.eval()
    
    with torch.no_grad():
        # 干净图像的流程
        clean_semantic = encoder(clean_images)
        clean_recon = decoder(clean_semantic)
        clean_outputs = classifier(clean_recon)
        clean_pred = clean_outputs.argmax(dim=1)
        clean_acc = (clean_pred == labels).float().mean().item()
        
        # 对抗样本的流程
        adv_semantic = encoder(adv_images)
        adv_recon = decoder(adv_semantic)
        adv_outputs = classifier(adv_recon)
        adv_pred = adv_outputs.argmax(dim=1)
        adv_acc = (adv_pred == labels).float().mean().item()
        
        # 攻击成功率
        attack_success_rate = (adv_pred != labels).float().mean().item()
    
    # 扰动统计
    perturbation = (adv_images - clean_images).cpu().numpy()
    
    # PSNR计算
    mse = np.mean((adv_images.cpu().numpy() - clean_images.cpu().numpy()) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    metrics = {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'attack_success_rate': attack_success_rate,
        'perturbation_l2_mean': np.mean(np.linalg.norm(perturbation.reshape(perturbation.shape[0], -1), ord=2, axis=1)),
        'perturbation_linf_max': np.max(np.abs(perturbation)),
        'psnr': psnr,
        'semantic_l2_distance': torch.norm(adv_semantic - clean_semantic, p=2).item() / clean_semantic.numel(),
    }
    
    return metrics


def calculate_psnr(img1, img2):
    """
    计算PSNR（峰值信噪比）
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2):
    """
    简化的SSIM计算（结构相似度）
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    
    sigma1_sq = torch.var(img1)
    sigma2_sq = torch.var(img2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim.item()


def print_metrics(metrics, title="Attack Evaluation"):
    """
    打印评估指标
    """
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")
    print(f"{'=' * 50}\n")


def visualize_attack(clean_images, adv_images, labels, pred_clean, pred_adv, num_samples=5, save_path=None):
    """
    可视化对抗攻击结果
    
    参数:
        clean_images: 干净图像
        adv_images: 对抗样本
        labels: 真实标签
        pred_clean: 干净图像的预测
        pred_adv: 对抗样本的预测
        num_samples: 显示样本数
        save_path: 保存路径
    """
    num_samples = min(num_samples, clean_images.size(0))
    
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
    
    for i in range(num_samples):
        # 干净图像
        clean_img = clean_images[i].cpu().squeeze().numpy()
        axes[0, i].imshow(clean_img, cmap='gray')
        axes[0, i].set_title(f'Clean\nTrue: {labels[i]}, Pred: {pred_clean[i]}')
        axes[0, i].axis('off')
        
        # 对抗样本
        adv_img = adv_images[i].cpu().squeeze().numpy()
        axes[1, i].imshow(adv_img, cmap='gray')
        axes[1, i].set_title(f'Adversarial\nTrue: {labels[i]}, Pred: {pred_adv[i]}')
        axes[1, i].axis('off')
        
        # 扰动（放大显示）
        perturbation = (adv_img - clean_img) * 10  # 放大10倍以便可视化
        axes[2, i].imshow(perturbation, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[2, i].set_title('Perturbation (×10)')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

