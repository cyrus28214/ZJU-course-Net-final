# 语义通信系统安全性探索

基于深度学习的语义通信系统对抗攻击研究项目，实现了MNIST手写数字识别的语义通信系统，并对其进行了全面的对抗攻击评估。

## 项目概述

本项目研究针对语义通信系统的对抗攻击方法，特别是针对基于深度学习的图像语义通信系统。语义通信是Beyond-5G通信的重要技术，通过传输语义信息而非原始比特来实现低延迟和高信噪比容忍度。然而，由于使用深度学习技术，语义通信系统可能容易受到对抗样本攻击。

### 主要成果

- ✅ 实现了完整的MNIST语义通信系统（3种压缩率：0.1, 0.2, 0.3）
- ✅ 实现了5种对抗攻击方法（FGSM、PGD、端到端、语义特征、信道噪声）
- ✅ 全面评估了系统脆弱性（PGD攻击达到100%成功率）
- ✅ 生成了详细的实验报告和可视化结果

## 环境要求

- Python 3.7+
- PyTorch 1.5.1+ (支持CUDA更佳)
- NumPy 1.21.2+
- Matplotlib 3.0.0+
- Pandas 1.0.0+
- SciPy 1.0.0+
- Pillow 8.0.0+

详细依赖见 [requirements.txt](./requirements.txt)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或安装PyTorch (CUDA版本):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install numpy matplotlib pandas scipy pillow
```

### 2. 训练模型

**训练分类器**:
```bash
cd semantic_extraction
python MLP_MNIST_model.py
```

**训练语义编码器**:
```bash
python MNIST.py
```

### 3. 运行对抗攻击测试

```bash
cd ..
python test_attack.py --compression-rate 0.1 --batch-size 100
```

## 实验结果

### 攻击成功率

**FGSM攻击**:
- ε=0.1: 15.00%
- ε=0.2: 64.00%
- ε=0.3: 92.00%

**PGD攻击**:
- ε=0.1: 31.00%
- ε=0.2: 98.00%
- ε=0.3: **100.00%**

**端到端攻击**:
- 攻击成功率: 24.00%（但图像质量最好，PSNR=20.86 dB）

详细结果见 [ATTACK_RESULTS_SUMMARY.md](./ATTACK_RESULTS_SUMMARY.md)

## 项目结构

```
.
├── semantic_extraction/          # 语义提取模块
│   ├── MLP_MNIST_model.py       # 分类器训练
│   ├── MNIST.py                 # 编码器训练
│   └── dataset/                 # 数据集目录
├── attacks/                      # 对抗攻击模块
│   ├── fgsm.py                  # FGSM攻击
│   ├── pgd.py                   # PGD攻击
│   ├── semantic_attack.py       # 语义空间攻击
│   └── evaluate.py              # 评估工具
├── semantic_system_with_DA/      # 数据适配模块
├── saved_model/                  # 训练好的模型（不上传）
├── test_attack.py               # 攻击测试脚本
├── 实验报告_语义通信安全性探索.md  # 完整实验报告
├── ATTACK_RESULTS_SUMMARY.md    # 攻击结果总结
└── requirements.txt             # 依赖列表
```

## 文档

- [实验报告](./实验报告_语义通信安全性探索.md) - 完整的实验报告（中文）
- [攻击结果总结](./ATTACK_RESULTS_SUMMARY.md) - 详细的攻击结果
- [攻击方法说明](./ATTACK_METHODS.md) - 攻击方法详细说明
- [GitHub上传指南](./GITHUB_UPLOAD_STEPS.md) - 如何上传到GitHub

## 引用

如果使用本代码，请引用原始论文：

```bibtex
@ARTICLE{9953099,
  author={Zhang, Hongwei and Shao, Shuo and Tao, Meixia and Bi, Xiaoyan and Letaief, Khaled B.},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={Deep Learning-Enabled Semantic Communication Systems With Task-Unaware Transmitter and Dynamic Data}, 
  year={2023},
  volume={41},
  number={1},
  pages={170-185},
  doi={10.1109/JSAC.2022.3221991}
}
```

## 许可证

见 [LICENSE](./LICENSE) 文件

## 致谢

基于原始代码库: https://github.com/SJTU-mxtao/Semantic-Communication-Systems
