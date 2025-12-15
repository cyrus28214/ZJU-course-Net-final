#!/usr/bin/env python
# encoding: utf-8
"""
Generate Visualization Charts for Attack Experiment Results
生成对抗攻击实验结果的可视化图表
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_results(json_path):
    """加载实验结果JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_attack_comparison(results, output_path):
    """生成攻击方法对比柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 提取数据 (使用epsilon=0.3的结果)
    methods = []
    success_rates = []
    
    # FGSM
    for item in results['fgsm']:
        if item['epsilon'] == 0.3:
            methods.append('FGSM')
            success_rates.append(item['metrics']['attack_success_rate'] * 100)
            break
    
    # PGD
    for item in results['pgd']:
        if item['epsilon'] == 0.3:
            methods.append('PGD')
            success_rates.append(item['metrics']['attack_success_rate'] * 100)
            break
    
    # End-to-End
    if results['e2e']:
        methods.append('End-to-End')
        success_rates.append(results['e2e']['metrics']['attack_success_rate'] * 100)
    
    colors = ['#FF6B6B', '#EE5A6F', '#4ECDC4']
    bars = ax.bar(methods, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Attack Methods Comparison (ε=0.3)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_epsilon_curves(results, output_path):
    """生成不同epsilon下的性能曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 提取FGSM数据
    fgsm_eps = [item['epsilon'] for item in results['fgsm']]
    fgsm_acc = [item['metrics']['adversarial_accuracy'] * 100 for item in results['fgsm']]
    fgsm_success = [item['metrics']['attack_success_rate'] * 100 for item in results['fgsm']]
    
    # 提取PGD数据
    pgd_eps = [item['epsilon'] for item in results['pgd']]
    pgd_acc = [item['metrics']['adversarial_accuracy'] * 100 for item in results['pgd']]
    pgd_success = [item['metrics']['attack_success_rate'] * 100 for item in results['pgd']]
    
    # 左图: 对抗准确率
    ax1.plot(fgsm_eps, fgsm_acc, 'o-', label='FGSM', linewidth=2, markersize=8, color='#FF6B6B')
    ax1.plot(pgd_eps, pgd_acc, 's-', label='PGD', linewidth=2, markersize=8, color='#EE5A6F')
    ax1.axhline(y=results['clean']['accuracy'] * 100, color='green', linestyle='--', 
                label='Clean Accuracy', linewidth=2)
    ax1.set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Adversarial Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Epsilon', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # 右图: 攻击成功率
    ax2.plot(fgsm_eps, fgsm_success, 'o-', label='FGSM', linewidth=2, markersize=8, color='#FF6B6B')
    ax2.plot(pgd_eps, pgd_success, 's-', label='PGD', linewidth=2, markersize=8, color='#EE5A6F')
    ax2.set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Attack Success vs Epsilon', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_psnr_tradeoff(results, output_path):
    """生成PSNR与攻击成功率的权衡图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 提取数据
    fgsm_psnr = [item['metrics']['psnr'] for item in results['fgsm']]
    fgsm_success = [item['metrics']['attack_success_rate'] * 100 for item in results['fgsm']]
    fgsm_eps = [item['epsilon'] for item in results['fgsm']]
    
    pgd_psnr = [item['metrics']['psnr'] for item in results['pgd']]
    pgd_success = [item['metrics']['attack_success_rate'] * 100 for item in results['pgd']]
    pgd_eps = [item['epsilon'] for item in results['pgd']]
    
    # 绘制散点图
    scatter1 = ax.scatter(fgsm_psnr, fgsm_success, s=200, c=fgsm_eps, 
                         cmap='Reds', marker='o', edgecolors='black', linewidth=1.5,
                         label='FGSM', alpha=0.8)
    scatter2 = ax.scatter(pgd_psnr, pgd_success, s=200, c=pgd_eps, 
                         cmap='Blues', marker='s', edgecolors='black', linewidth=1.5,
                         label='PGD', alpha=0.8)
    
    # 添加epsilon标签
    for i, eps in enumerate(fgsm_eps):
        ax.annotate(f'ε={eps}', (fgsm_psnr[i], fgsm_success[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    for i, eps in enumerate(pgd_eps):
        ax.annotate(f'ε={eps}', (pgd_psnr[i], pgd_success[i]), 
                   xytext=(5, -15), textcoords='offset points', fontsize=9)
    
    # End-to-End点
    if results['e2e']:
        e2e_psnr = results['e2e']['metrics']['psnr']
        e2e_success = results['e2e']['metrics']['attack_success_rate'] * 100
        ax.scatter([e2e_psnr], [e2e_success], s=250, c='#4ECDC4', 
                  marker='^', edgecolors='black', linewidth=2,
                  label='End-to-End', alpha=0.9)
        ax.annotate('E2E', (e2e_psnr, e2e_success), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('PSNR (dB) - Higher is Better', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Success Rate (%) - Higher is Worse', fontsize=12, fontweight='bold')
    ax.set_title('Attack-Stealth Tradeoff: PSNR vs Success Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_perturbation_comparison(results, output_path):
    """生成扰动大小对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准备数据
    epsilons = [item['epsilon'] for item in results['fgsm']]
    
    fgsm_l2 = [item['metrics']['perturbation_l2_mean'] for item in results['fgsm']]
    pgd_l2 = [item['metrics']['perturbation_l2_mean'] for item in results['pgd']]
    
    fgsm_linf = [item['metrics']['perturbation_linf_max'] for item in results['fgsm']]
    pgd_linf = [item['metrics']['perturbation_linf_max'] for item in results['pgd']]
    
    x = np.arange(len(epsilons))
    width = 0.35
    
    # L2扰动
    bars1 = ax1.bar(x - width/2, fgsm_l2, width, label='FGSM', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, pgd_l2, width, label='PGD', color='#EE5A6F', alpha=0.8)
    
    ax1.set_xlabel('Epsilon', fontsize=12, fontweight='bold')
    ax1.set_ylabel('L2 Perturbation', fontsize=12, fontweight='bold')
    ax1.set_title('L2 Perturbation Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{eps}' for eps in epsilons])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # L∞扰动
    bars3 = ax2.bar(x - width/2, fgsm_linf, width, label='FGSM', color='#FF6B6B', alpha=0.8)
    bars4 = ax2.bar(x + width/2, pgd_linf, width, label='PGD', color='#EE5A6F', alpha=0.8)
    
    ax2.set_xlabel('Epsilon', fontsize=12, fontweight='bold')
    ax2.set_ylabel('L∞ Perturbation', fontsize=12, fontweight='bold')
    ax2.set_title('L∞ Perturbation Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{eps}' for eps in epsilons])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_charts(json_path, output_dir):
    """生成所有图表"""
    print("Loading experiment results...")
    results = load_results(json_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating charts...")
    plot_attack_comparison(results, output_dir / 'attack_comparison.png')
    plot_epsilon_curves(results, output_dir / 'epsilon_curves.png')
    plot_psnr_tradeoff(results, output_dir / 'psnr_tradeoff.png')
    plot_perturbation_comparison(results, output_dir / 'perturbation_comparison.png')
    
    print(f"\nAll charts saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate visualization charts from experiment results')
    parser.add_argument('--input', type=str, default='results/experiment_results.json',
                       help='Path to experiment results JSON file')
    parser.add_argument('--output', type=str, default='images',
                       help='Output directory for charts')
    args = parser.parse_args()
    
    generate_all_charts(args.input, args.output)


if __name__ == '__main__':
    main()
