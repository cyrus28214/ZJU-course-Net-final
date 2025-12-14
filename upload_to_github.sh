#!/bin/bash
# GitHub上传脚本 (Linux/Mac/Git Bash)
# 使用方法: bash upload_to_github.sh

echo "========================================"
echo "GitHub上传脚本"
echo "========================================"
echo ""

# 检查Git是否安装
if ! command -v git &> /dev/null; then
    echo "[错误] Git未安装，请先安装Git"
    exit 1
fi

echo "[1/7] 检查Git状态..."
if [ -d .git ]; then
    echo "Git仓库已存在"
else
    echo "初始化Git仓库..."
    git init
fi

echo ""
echo "[2/7] 检查.gitignore文件..."
if [ -f .gitignore ]; then
    echo ".gitignore文件存在"
else
    echo "[警告] .gitignore文件不存在"
fi

echo ""
echo "[3/7] 添加文件到Git..."
git add .

echo ""
echo "[4/7] 检查将要提交的文件..."
git status

echo ""
echo "[5/7] 提交文件..."
git commit -m "Initial commit: Semantic Communication System with Adversarial Attacks

- Implemented semantic communication system for MNIST
- Added FGSM, PGD, and end-to-end adversarial attacks
- Complete experimental results and documentation
- All code tested and reproducible"

echo ""
echo "[6/7] 连接到远程仓库..."
git remote remove origin 2>/dev/null
git remote add origin https://github.com/cyrus28214/ZJU-course-Net-final.git

echo ""
echo "[7/7] 设置主分支并推送..."
git branch -M main
git push -u origin main

echo ""
echo "========================================"
echo "上传完成！"
echo "========================================"
echo ""
echo "如果遇到认证问题，请："
echo "1. 使用GitHub Personal Access Token代替密码"
echo "2. 或配置SSH密钥"
echo ""

