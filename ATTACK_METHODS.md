# 语义通信系统对抗攻击方法研究

## 项目背景

语义通信系统使用深度学习技术来提取和传输语义特征，相比传统通信系统具有低延迟和高信噪比容忍度的优势。然而，深度学习模型对对抗样本的脆弱性也为语义通信系统带来了新的安全威胁。

## 系统架构分析

根据代码分析，本语义通信系统包含以下组件：

### 1. 编码器（Encoder）
- **位置**: `semantic_extraction/MNIST.py` 中的 `MLP` 类
- **功能**: 将原始图像（28×28=784维）编码为低维语义特征（压缩率0.1-0.3，即78-235维）
- **结构**: 
  - FC1: 784 → 1024
  - FC2: 1024 → channel（压缩后的维度）
  - 添加信道噪声（SNR=10dB）

### 2. 解码器（Decoder）
- **位置**: 同一 `MLP` 类的后半部分
- **功能**: 从噪声语义特征恢复原始图像
- **结构**:
  - FC1: channel → 1024
  - FC2: 1024 → 784

### 3. 分类器（Classifier）
- **位置**: `MLP_MNIST` 类
- **功能**: 对恢复的图像进行分类（10类MNIST数字）
- **结构**: 784 → 500 → 250 → 125 → 10

## 攻击目标

### 目标1: 白盒攻击
- **场景**: 攻击者完全了解模型架构和参数
- **目标**: 生成对抗样本，使得：
  1. 原始图像和对抗样本在视觉上相似
  2. 对抗样本经过编码-解码后，分类器识别错误
  3. 对抗样本能成功传输通过信道

### 目标2: 黑盒攻击
- **场景**: 攻击者只能访问模型的输入输出
- **目标**: 通过查询模型，生成对抗样本

## 潜在攻击方法

### 方法1: 端到端对抗攻击（End-to-End Attack）

**攻击流程**:
```
原始图像 → 对抗扰动 → 对抗样本 → 编码器 → 语义特征 → 信道噪声 → 解码器 → 恢复图像 → 分类器
```

**攻击目标函数**:
```python
loss = α * classification_loss(decoder(encoder(adv_image)), wrong_label) 
     + β * reconstruction_loss(adv_image, original_image)
     + γ * perturbation_norm(adv_image - original_image)
```

**优点**:
- 考虑完整的编码-解码-分类流程
- 攻击针对最终分类任务

**挑战**:
- 需要处理信道噪声的不确定性
- 优化过程复杂

### 方法2: 语义特征空间攻击

**攻击流程**:
```
原始图像 → 编码器 → 语义特征 → 对抗扰动 → 扰动语义特征 → 解码器 → 恢复图像 → 分类器
```

**攻击目标**:
- 在语义特征空间中添加小的扰动
- 使得解码后的图像分类错误

**优点**:
- 直接在低维语义空间操作
- 扰动可能更小

### 方法3: 信道对抗攻击

**攻击流程**:
```
语义特征 → 对抗噪声注入 → 扰动语义特征 → 解码器 → 恢复图像 → 分类器
```

**场景**: 攻击者在信道中注入对抗噪声

**优点**:
- 更贴近实际攻击场景
- 不需要修改发送端

### 方法4: 基于梯度的攻击方法

#### 4.1 FGSM (Fast Gradient Sign Method)
- 快速生成对抗样本
- 适合初步验证攻击可行性

#### 4.2 PGD (Projected Gradient Descent)
- 迭代优化，攻击成功率更高
- 可以添加约束保证扰动在感知上不可见

#### 4.3 C&W (Carlini & Wagner)
- 强大的白盒攻击方法
- 可以精确控制扰动的L2或L∞范数

### 方法5: 基于优化的攻击方法

**优化问题**:
```
minimize: ||x_adv - x_original||_p
subject to: argmax(classifier(decoder(encoder(x_adv)))) ≠ correct_label
```

使用投影梯度下降或Adam优化器求解。

## 实现计划

### 阶段1: 白盒攻击实现 ✅ 已完成
1. **分析代码结构** ✅
   - 已完成：分析编码器、解码器、分类器结构
   
2. **实现基础对抗攻击** ✅
   - ✅ 实现FGSM攻击 (`attacks/fgsm.py`)
   - ✅ 实现PGD攻击 (`attacks/pgd.py`)
   - ⏳ 测试攻击成功率（等待模型训练完成）

3. **端到端攻击** ✅
   - ✅ 实现考虑编码-解码-分类完整流程的攻击 (`attacks/semantic_attack.py`)
   - ✅ 实现语义特征空间攻击
   - ✅ 实现信道噪声攻击
   - ⏳ 评估不同压缩率下的攻击效果（等待模型训练完成）

4. **评估工具** ✅
   - ✅ 实现攻击效果评估 (`attacks/evaluate.py`)
   - ✅ 实现可视化工具
   - ✅ 创建测试脚本 (`test_attack.py`)

### 阶段2: 黑盒攻击实现
1. **模型查询攻击**
   - 实现基于查询的对抗样本生成
   - 使用替代模型进行迁移攻击

2. **无需查询攻击**
   - 研究通用对抗扰动
   - 研究对抗补丁

### 阶段3: 防御方法研究
1. **对抗训练**
   - 在训练过程中加入对抗样本
   
2. **输入变换**
   - 随机化输入
   - 输入预处理

3. **特征压缩防御**
   - 研究压缩率对鲁棒性的影响

## 实验评估指标

1. **攻击成功率 (Attack Success Rate)**
   - 成功误导分类器的对抗样本比例

2. **扰动大小 (Perturbation Size)**
   - L2范数: ||x_adv - x_original||_2
   - L∞范数: ||x_adv - x_original||_∞

3. **视觉相似度 (Visual Similarity)**
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)

4. **信道传输效果**
   - 不同SNR下的攻击成功率
   - 对抗样本在信道中的鲁棒性

## 预期挑战

1. **信道噪声的影响**
   - 信道噪声可能破坏精心设计的对抗扰动
   - 需要研究鲁棒的对抗样本生成方法

2. **压缩的影响**
   - 不同的压缩率可能影响攻击效果
   - 需要在不同压缩率下测试攻击

3. **端到端优化**
   - 包含编码、解码、分类的端到端优化较复杂
   - 可能需要特殊的优化技巧

## 已实现的攻击方法

### 1. FGSM攻击 (`attacks/fgsm.py`)
**原理**: 使用梯度的符号来生成对抗扰动
```python
perturbation = epsilon * images.grad.sign()
adv_images = images + perturbation
```

**特点**:
- 快速，单步攻击
- 计算效率高
- 适合初步测试攻击可行性

**实现**:
- `fgsm_attack()`: 基础FGSM攻击
- `fgsm_attack_semantic()`: 针对语义通信系统的FGSM攻击

### 2. PGD攻击 (`attacks/pgd.py`)
**原理**: 多步迭代优化，每步投影回epsilon球内
```python
for i in range(num_iter):
    grad = compute_gradient(adv_images)
    adv_images = adv_images + alpha * grad.sign()
    perturbation = clip(adv_images - original, -epsilon, epsilon)
```

**特点**:
- 强大的白盒攻击
- 迭代优化，成功率更高
- 可控制扰动大小

**实现**:
- `pgd_attack()`: 基础PGD攻击
- `pgd_attack_semantic()`: 针对语义通信系统的PGD攻击

### 3. 端到端攻击 (`attacks/semantic_attack.py`)
**原理**: 优化整个编码-解码-分类流程
```python
loss = lambda_class * classification_loss - lambda_recon * reconstruction_loss
```

**特点**:
- 考虑完整的语义通信流程
- 平衡攻击效果和图像质量
- 更针对性的攻击

**实现**:
- `end_to_end_attack()`: 端到端对抗攻击
- `semantic_feature_attack()`: 语义特征空间攻击
- `channel_noise_attack()`: 信道噪声攻击

### 4. 评估工具 (`attacks/evaluate.py`)
**功能**:
- 攻击成功率计算
- 扰动统计 (L2, L∞范数)
- PSNR、SSIM计算
- 可视化对抗样本
- 语义特征距离分析

## 使用方法

### 快速测试
训练完成后运行：
```bash
python test_attack.py
```

该脚本会：
1. 加载训练好的模型
2. 对测试集进行FGSM、PGD、端到端攻击
3. 评估攻击效果
4. 生成可视化结果

### 自定义攻击
```python
from attacks import fgsm_attack_semantic, evaluate_semantic_attack

# 生成对抗样本
adv_images, perturbation = fgsm_attack_semantic(
    encoder, decoder, classifier,
    images, labels, epsilon=0.3
)

# 评估攻击效果
metrics = evaluate_semantic_attack(
    encoder, decoder, classifier,
    images, adv_images, labels
)
```

## 下一步工作

1. ⏳ 等待语义提取模型训练完成
2. 🔜 运行 `test_attack.py` 测试攻击效果
3. 🔜 测试攻击在不同压缩率（0.1, 0.2, 0.3）下的效果
4. 🔜 测试攻击在不同SNR下的鲁棒性
5. 🔜 分析攻击的可转移性
6. 🔜 探索防御方法（对抗训练、输入变换等）

