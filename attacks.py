#!/usr/bin/env python
# encoding: utf-8
"""
Adversarial Attack Testing on Semantic Communication System
对语义通信系统进行对抗攻击测试 (重构版)
"""

import os
import sys
import random
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader

# 添加路径
sys.path.append('semantic_extraction')

# 导入模块化组件
from models import MLP_MNIST, MLP_Encoder_Decoder, SemanticCommSystem
from experiment_runner import AttackExperiment


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Test adversarial attacks on semantic communication system')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch-size', type=int, default=100, help='test batch size')
    parser.add_argument('--compression-rate', type=float, default=0.1, help='compression rate')
    parser.add_argument('--device', type=str, default=None, help='device (cuda/cpu)')
    parser.add_argument('--model-dir', type=str, default='saved_model', help='model directory')
    parser.add_argument('--output-dir', type=str, default='results', help='output directory')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.1, 0.2, 0.3], help='epsilon values')
    parser.add_argument('--pgd-alpha', type=float, default=0.01, help='PGD step size')
    parser.add_argument('--pgd-iter', type=int, default=40, help='PGD iterations')
    return parser.parse_args()


def setup_environment(args):
    """设置运行环境"""
    # 配置logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 确定设备
    device = torch.device(
        args.device if args.device 
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    logging.info(f'使用设备: {device}')
    
    return device


def data_transform(x):
    """数据预处理"""
    x = np.array(x, dtype='float32') / 255
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x


def load_data(args):
    """加载MNIST数据"""
    logging.info('正在加载MNIST数据集...')
    testset = mnist.MNIST(
        './semantic_extraction/dataset/mnist',
        train=False,
        transform=data_transform,
        download=True
    )
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    # 获取一批测试数据
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    
    return images, labels


def load_models(args, device):
    """加载训练好的模型"""
    logging.info('正在加载模型...')
    
    MODEL_DIR = Path(args.model_dir)
    channel = int(args.compression_rate * 28 * 28)
    
    # 加载编码器-解码器
    coder = MLP_Encoder_Decoder(channel=channel).to(device)
    coder_path = MODEL_DIR / f'MLP_MNIST_coder_{args.compression_rate:.6f}.pkl'
    
    if not coder_path.exists():
        logging.error(f'编码器模型未找到: {coder_path}')
        logging.error('请先运行训练脚本')
        sys.exit(1)
    
    coder.load_state_dict(torch.load(str(coder_path), map_location=device))
    coder.eval()
    logging.info(f'已加载编码器: {coder_path}')
    
    # 加载分类器
    classifier = MLP_MNIST().to(device)
    classifier_path = MODEL_DIR / 'MLP_MNIST.pkl'
    
    if not classifier_path.exists():
        logging.error(f'分类器模型未找到: {classifier_path}')
        sys.exit(1)
    
    classifier.load_state_dict(torch.load(str(classifier_path), map_location=device))
    classifier.eval()
    logging.info(f'已加载分类器: {classifier_path}')
    
    # 创建端到端模型
    full_model = SemanticCommSystem(coder, classifier).to(device)
    
    return coder, classifier, full_model


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置环境
    device = setup_environment(args)
    
    # 加载数据
    images, labels = load_data(args)
    images = images.to(device)
    labels = labels.to(device)
    
    # 加载模型
    coder, classifier, full_model = load_models(args, device)
    
    # 创建实验管理器
    experiment = AttackExperiment(
        encoder=coder.encode,
        decoder=coder.decode,
        classifier=classifier,
        full_model=full_model,
        device=device
    )
    
    # 打印实验信息
    logging.info('\n' + '='*60)
    logging.info('语义通信系统对抗攻击实验')
    logging.info('='*60)
    
    # 1. 测试干净样本准确率
    clean_acc = experiment.test_clean_accuracy(images, labels)
    
    # 2. FGSM攻击
    adv_images_fgsm = experiment.run_fgsm_attack(images, labels, args.epsilons)
    
    # 3. PGD攻击
    adv_images_pgd = experiment.run_pgd_attack(
        images, labels, args.epsilons,
        alpha=args.pgd_alpha,
        num_iter=args.pgd_iter
    )
    
    # 4. 端到端攻击
    adv_images_e2e = experiment.run_e2e_attack(
        images, labels,
        epsilon=args.epsilons[-1] if args.epsilons else 0.3,
        alpha=args.pgd_alpha,
        num_iter=args.pgd_iter
    )
    
    # 5. 可视化结果
    experiment.visualize_results(
        images, labels,
        adv_images_fgsm, adv_images_pgd,
        args.output_dir
    )
    
    # 6. 保存结果
    output_path = Path(args.output_dir) / 'experiment_results.json'
    experiment.save_results(output_path)
    
if __name__ == '__main__':
    main()
