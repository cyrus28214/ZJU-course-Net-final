#!/usr/bin/env python
# encoding: utf-8
"""
Adversarial Attack Experiment Runner
模块化的对抗攻击实验运行器
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack_semantic
from attacks.semantic_attack import end_to_end_attack
from attacks.evaluate import evaluate_semantic_attack, visualize_attack


class AttackExperiment:
    """对抗攻击实验管理器
    
    负责运行不同类型的对抗攻击并收集结果
    """
    
    def __init__(self, encoder, decoder, classifier, full_model, device):
        """
        Args:
            encoder: 语义编码器
            decoder: 语义解码器
            classifier: 分类器
            full_model: 端到端模型
            device: 运行设备
        """
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.full_model = full_model
        self.device = device
        self.results = {
            'clean': {},
            'fgsm': [],
            'pgd': [],
            'e2e': {}
        }
    
    def test_clean_accuracy(self, images, labels):
        """测试干净样本准确率"""
        logging.info('\n1. Testing Clean Accuracy...')
        with torch.no_grad():
            semantic = self.encoder(images)
            reconstructed = self.decoder(semantic)
            outputs = self.classifier(reconstructed)
            pred = outputs.argmax(dim=1)
            clean_acc = (pred == labels).float().mean().item()
            
        self.results['clean'] = {
            'accuracy': clean_acc,
            'num_samples': len(images)
        }
        
        logging.info(f"Clean Accuracy: {clean_acc:.4f}")
        return clean_acc
    
    def run_fgsm_attack(self, images, labels, epsilons):
        """运行FGSM攻击"""
        logging.info('\n' + '='*60)
        logging.info('2. FGSM Attack')
        logging.info('='*60)
        
        adv_images_fgsm = None
        for epsilon in epsilons:
            logging.info(f"\nEpsilon = {epsilon}")
            
            adv_images_fgsm, perturbation = fgsm_attack(
                self.full_model, images, labels, epsilon=epsilon
            )
            
            metrics = evaluate_semantic_attack(
                self.encoder, self.decoder, self.classifier,
                images, adv_images_fgsm, labels, self.device
            )
            
            result = {
                'epsilon': epsilon,
                'metrics': metrics
            }
            self.results['fgsm'].append(result)
            
            self._print_metrics(metrics, f"FGSM Attack (ε={epsilon})")
        
        return adv_images_fgsm
    
    def run_pgd_attack(self, images, labels, epsilons, alpha=0.01, num_iter=40):
        """运行PGD攻击"""
        logging.info("\n" + "="*60)
        logging.info("3. PGD Attack")
        logging.info("="*60)
        
        adv_images_pgd = None
        for epsilon in epsilons:
            logging.info(f"\nEpsilon = {epsilon}")
            
            adv_images_pgd, perturbation = pgd_attack_semantic(
                self.encoder, self.decoder, self.classifier,
                images, labels,
                epsilon=epsilon, alpha=alpha, num_iter=num_iter
            )
            
            metrics = evaluate_semantic_attack(
                self.encoder, self.decoder, self.classifier,
                images, adv_images_pgd, labels, self.device
            )
            
            result = {
                'epsilon': epsilon,
                'alpha': alpha,
                'num_iter': num_iter,
                'metrics': metrics
            }
            self.results['pgd'].append(result)
            
            self._print_metrics(metrics, f"PGD Attack (ε={epsilon})")
        
        return adv_images_pgd
    
    def run_e2e_attack(self, images, labels, epsilon=0.3, alpha=0.01, num_iter=40):
        """运行端到端攻击"""
        logging.info("\n" + "="*60)
        logging.info("4. End-to-End Attack")
        logging.info("="*60)
        
        adv_images_e2e, perturbation, metrics_e2e = end_to_end_attack(
            self.encoder, self.decoder, self.classifier,
            images, labels,
            epsilon=epsilon, alpha=alpha, num_iter=num_iter
        )
        
        metrics = evaluate_semantic_attack(
            self.encoder, self.decoder, self.classifier,
            images, adv_images_e2e, labels, self.device
        )
        
        self.results['e2e'] = {
            'epsilon': epsilon,
            'alpha': alpha,
            'num_iter': num_iter,
            'metrics': metrics,
            'history': metrics_e2e['attack_success_history']
        }
        
        self._print_metrics(metrics, "End-to-End Attack")
        logging.info(f"Attack Success Rate History: {metrics_e2e['attack_success_history'][-5:]}")
        
        return adv_images_e2e
    
    def visualize_results(self, images, labels, adv_images_fgsm, adv_images_pgd, output_dir):
        """可视化攻击结果"""
        logging.info("\n5. Generating Visualizations...")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 可视化FGSM攻击
        if adv_images_fgsm is not None:
            with torch.no_grad():
                semantic = self.encoder(adv_images_fgsm[:5])
                reconstructed = self.decoder(semantic)
                pred_clean = self.classifier(self.decoder(self.encoder(images[:5]))).argmax(dim=1)
                pred_adv = self.classifier(reconstructed).argmax(dim=1)
            
            visualize_attack(
                images[:5].view(-1, 1, 28, 28),
                adv_images_fgsm[:5].view(-1, 1, 28, 28),
                labels[:5].cpu(),
                pred_clean.cpu(),
                pred_adv.cpu(),
                num_samples=5,
                save_path=str(output_dir / 'attack_results_fgsm.png')
            )
        
        # 可视化PGD攻击
        if adv_images_pgd is not None:
            with torch.no_grad():
                semantic = self.encoder(adv_images_pgd[:5])
                reconstructed = self.decoder(semantic)
                pred_clean = self.classifier(self.decoder(self.encoder(images[:5]))).argmax(dim=1)
                pred_adv = self.classifier(reconstructed).argmax(dim=1)
            
            visualize_attack(
                images[:5].view(-1, 1, 28, 28),
                adv_images_pgd[:5].view(-1, 1, 28, 28),
                labels[:5].cpu(),
                pred_clean.cpu(),
                pred_adv.cpu(),
                num_samples=5,
                save_path=str(output_dir / 'attack_results_pgd.png')
            )
    
    def save_results(self, output_path):
        """保存实验结果到JSON文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化格式
        serializable_results = self._make_serializable(self.results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logging.info(f'\nSaved experiment results to {output_path}')
    
    def _make_serializable(self, obj):
        """递归转换为可JSON序列化的格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _print_metrics(self, metrics, title):
        """打印评估指标"""
        logging.info(f"\n{'='*60}")
        logging.info(f"{title}")
        logging.info(f"{'='*60}")
        logging.info(f"Adversarial Accuracy: {metrics['adversarial_accuracy']:.4f}")
        logging.info(f"Attack Success Rate: {metrics['attack_success_rate']:.4f}")
        logging.info(f"L2 Perturbation: {metrics['perturbation_l2_mean']:.4f}")
        logging.info(f"L∞ Perturbation: {metrics['perturbation_linf_max']:.4f}")
        logging.info(f"PSNR: {metrics['psnr']:.2f} dB")
        logging.info(f"Semantic Distance: {metrics['semantic_l2_distance']:.4f}")

