#!/usr/bin/env python
# encoding: utf-8
"""
测试对抗攻击
对语义通信系统进行对抗攻击测试
"""

import os
from pathlib import Path
import sys
import random
import argparse
import json
from datetime import datetime
import logging
import torch
import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 添加路径
sys.path.append('semantic_extraction')
sys.path.append('attacks')

from attacks.fgsm import fgsm_attack, fgsm_attack_semantic
from attacks.pgd import pgd_attack, pgd_attack_semantic
from attacks.semantic_attack import end_to_end_attack, semantic_feature_attack
from attacks.evaluate import evaluate_semantic_attack, print_metrics, visualize_attack


def parse_args():
    parser = argparse.ArgumentParser(description='Test adversarial attacks on semantic comm system')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch-size', type=int, default=100, help='test batch size')
    parser.add_argument('--compression-rate', type=float, default=0.1, help='compression rate to compute channel dim')
    parser.add_argument('--device', type=str, default=None, help='device to use, e.g. cuda or cpu')
    parser.add_argument('--model-dir', type=str, default='saved_model', help='directory containing saved models')
    parser.add_argument('--coder-name', type=str, default=None, help='encoder-decoder model filename (default derived from compression rate)')
    parser.add_argument('--classifier-name', type=str, default='MLP_MNIST.pkl', help='classifier model filename')
    parser.add_argument('--output-dir', type=str, default='results', help='directory to save visualizations and outputs')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.1, 0.2, 0.3], help='list of epsilons for FGSM/PGD attacks')
    parser.add_argument('--pgd-alpha', type=float, default=0.01, help='PGD step size')
    parser.add_argument('--pgd-iter', type=int, default=40, help='PGD number of iterations')
    parser.add_argument('--e2e-alpha', type=float, default=0.01, help='end-to-end attack step size')
    parser.add_argument('--e2e-iter', type=int, default=40, help='end-to-end attack iterations')
    parser.add_argument('--save-config', action='store_true', help='save run configuration to output dir as JSON')
    return parser.parse_args()


args = parse_args()

# logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 固定随机种子
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 检测设备
device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
logging.info(f'Using device: {device}')

# 数据预处理
def data_transform(x):
    x = np.array(x, dtype='float32') / 255
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x

# 加载数据
logging.info('Loading MNIST dataset...')
testset = mnist.MNIST('./semantic_extraction/dataset/mnist', train=False, transform=data_transform, download=True)
test_data = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# prepare model and output paths
MODEL_DIR = Path(args.model_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 定义模型
class MLP_Encoder_Decoder(torch.nn.Module):
    """编码器-解码器"""
    def __init__(self, channel=78):
        super(MLP_Encoder_Decoder, self).__init__()
        # 编码器
        self.fc1_1 = torch.nn.Linear(28 * 28, 1024)
        self.fc1_2 = torch.nn.Linear(1024, channel)
        # 解码器
        self.fc2_1 = torch.nn.Linear(channel, 1024)
        self.fc2_2 = torch.nn.Linear(1024, 28 * 28)
    
    def encode(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc1_2(x))
        return x
    
    def decode(self, x):
        x = F.relu(self.fc2_1(x))
        x = F.relu(self.fc2_2(x))
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class MLP_MNIST(torch.nn.Module):
    """分类器"""
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

# 加载模型
logging.info('Loading models...')
compression_rate = args.compression_rate
channel = int(compression_rate * 28 * 28)

# 编码器-解码器
mlp_coder = MLP_Encoder_Decoder(channel=channel).to(device)
# determine coder filename
if args.coder_name:
    coder_name = args.coder_name
else:
    coder_name = f'MLP_MNIST_coder_{compression_rate:.6f}.pkl'

coder_path = MODEL_DIR / coder_name
if coder_path.exists():
    mlp_coder.load_state_dict(torch.load(str(coder_path), map_location=device))
    logging.info(f'Loaded coder from {coder_path}')
else:
    logging.error(f'Coder model not found at {coder_path}')
    logging.error('Please run training first and ensure the model is saved in the model directory.')
    sys.exit(1)

# 分类器
mlp_mnist = MLP_MNIST().to(device)
classifier_path = MODEL_DIR / args.classifier_name
if classifier_path.exists():
    mlp_mnist.load_state_dict(torch.load(str(classifier_path), map_location=device))
    logging.info(f'Loaded classifier from {classifier_path}')
else:
    logging.error(f'Classifier model not found at {classifier_path}')
    sys.exit(1)

mlp_coder.eval()
mlp_mnist.eval()

logging.info('\n' + '='*60)
logging.info('Testing Adversarial Attacks on Semantic Communication System')
logging.info('='*60)

# 获取一批测试数据
test_iter = iter(test_data)
images, labels = next(test_iter)
images = images.to(device)
labels = labels.to(device)

# 测试原始准确率
logging.info('\n1. Testing Clean Accuracy...')
with torch.no_grad():
    semantic = mlp_coder.encode(images)
    reconstructed = mlp_coder.decode(semantic)
    outputs = mlp_mnist(reconstructed)
    pred = outputs.argmax(dim=1)
    clean_acc = (pred == labels).float().mean().item()
    print(f"Clean Accuracy: {clean_acc:.4f}")

# ============ FGSM攻击 ============
logging.info('\n' + '='*60)
logging.info('2. FGSM Attack')
logging.info('='*60)

# 定义完整的模型（在循环外定义一次）
class SemanticCommSystem(torch.nn.Module):
    def __init__(self, encoder_decoder, classifier):
        super().__init__()
        self.coder = encoder_decoder
        self.classifier = classifier
    
    def forward(self, x):
        semantic = self.coder.encode(x)
        reconstructed = self.coder.decode(semantic)
        outputs = self.classifier(reconstructed)
        return outputs

full_model = SemanticCommSystem(mlp_coder, mlp_mnist).to(device)

adv_images_fgsm = None
for epsilon in args.epsilons:
    logging.info(f"\nEpsilon = {epsilon}")
    
    adv_images_fgsm, perturbation_fgsm = fgsm_attack(
        full_model, images, labels, epsilon=epsilon
    )
    
    # 评估
    metrics = evaluate_semantic_attack(
        mlp_coder.encode, mlp_coder.decode, mlp_mnist,
        images, adv_images_fgsm, labels, device
    )
    print_metrics(metrics, f"FGSM Attack (ε={epsilon})")

# ============ PGD攻击 ============
logging.info("\n" + "="*60)
logging.info("3. PGD Attack")
logging.info("="*60)

adv_images_pgd = None
for epsilon in args.epsilons:
    logging.info(f"\nEpsilon = {epsilon}")
    
    adv_images_pgd, perturbation_pgd = pgd_attack_semantic(
        mlp_coder.encode, mlp_coder.decode, mlp_mnist,
        images, labels,
        epsilon=epsilon, alpha=args.pgd_alpha, num_iter=args.pgd_iter
    )
    
    # 评估
    metrics = evaluate_semantic_attack(
        mlp_coder.encode, mlp_coder.decode, mlp_mnist,
        images, adv_images_pgd, labels, device
    )
    print_metrics(metrics, f"PGD Attack (ε={epsilon})")

# ============ 端到端攻击 ============
logging.info("\n" + "="*60)
logging.info("4. End-to-End Attack")
logging.info("="*60)

adv_images_e2e, perturbation_e2e, metrics_e2e = end_to_end_attack(
    mlp_coder.encode, mlp_coder.decode, mlp_mnist,
    images, labels,
    epsilon=args.epsilons[-1] if args.epsilons else 0.3, alpha=args.e2e_alpha, num_iter=args.e2e_iter
)

metrics = evaluate_semantic_attack(
    mlp_coder.encode, mlp_coder.decode, mlp_mnist,
    images, adv_images_e2e, labels, device
)
print_metrics(metrics, "End-to-End Attack")
print(f"Attack Success Rate History: {metrics_e2e['attack_success_history'][-5:]}")

# ============ 可视化 ============
logging.info("\n5. Generating Visualizations...")

# 可视化FGSM攻击
with torch.no_grad():
    semantic = mlp_coder.encode(adv_images_fgsm[:5])
    reconstructed = mlp_coder.decode(semantic)
    pred_clean = mlp_mnist(mlp_coder.decode(mlp_coder.encode(images[:5]))).argmax(dim=1)
    pred_adv = mlp_mnist(reconstructed).argmax(dim=1)

if adv_images_fgsm is not None:
    visualize_attack(
        images[:5].view(-1, 1, 28, 28),
        adv_images_fgsm[:5].view(-1, 1, 28, 28),
        labels[:5].cpu(),
        pred_clean.cpu(),
        pred_adv.cpu(),
        num_samples=5,
        save_path=str(OUTPUT_DIR / 'attack_results_fgsm.png')
    )
else:
    logging.warning("FGSM attack images not available for visualization")

# 可视化PGD攻击
if adv_images_pgd is not None:
    with torch.no_grad():
        semantic = mlp_coder.encode(adv_images_pgd[:5])
        reconstructed = mlp_coder.decode(semantic)
        pred_adv = mlp_mnist(reconstructed).argmax(dim=1)

    visualize_attack(
        images[:5].view(-1, 1, 28, 28),
        adv_images_pgd[:5].view(-1, 1, 28, 28),
        labels[:5].cpu(),
        pred_clean.cpu(),
        pred_adv.cpu(),
        num_samples=5,
        save_path=str(OUTPUT_DIR / 'attack_results_pgd.png')
    )
else:
    logging.warning("PGD attack images not available for visualization")

logging.info("\n" + "="*60)
logging.info("Attack Testing Complete!")
logging.info("="*60)
logging.info("\nSummary:")
logging.info(f"- Clean Accuracy: {clean_acc:.4f}")
logging.info(f"- FGSM Attack Success Rate (ε=0.3): Check above results")
logging.info(f"- PGD Attack Success Rate (ε=0.3): Check above results")
logging.info(f"- Visualizations saved to {OUTPUT_DIR}/attack_results_*.png")

# save run configuration
if args.save_config:
    cfg = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'seed': args.seed,
        'device': str(device),
        'model_dir': str(MODEL_DIR),
        'coder': str(coder_path),
        'classifier': str(classifier_path),
        'compression_rate': compression_rate,
        'epsilons': args.epsilons,
        'pgd_alpha': args.pgd_alpha,
        'pgd_iter': args.pgd_iter,
        'e2e_alpha': args.e2e_alpha,
        'e2e_iter': args.e2e_iter,
    }
    cfg_path = OUTPUT_DIR / f'run_config_{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.json'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    logging.info(f'Saved run configuration to {cfg_path}')

