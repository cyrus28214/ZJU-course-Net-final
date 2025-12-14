@echo off
REM GitHub上传脚本
REM 使用方法: 在项目根目录运行此脚本

echo ========================================
echo GitHub上传脚本
echo ========================================
echo.

REM 检查Git是否安装
git --version >nul 2>&1
if errorlevel 1 (
    echo [错误] Git未安装，请先安装Git
    echo 下载地址: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo [1/7] 检查Git状态...
if exist .git (
    echo Git仓库已存在
) else (
    echo 初始化Git仓库...
    git init
)

echo.
echo [2/7] 检查.gitignore文件...
if exist .gitignore (
    echo .gitignore文件存在
) else (
    echo [警告] .gitignore文件不存在
)

echo.
echo [3/7] 添加文件到Git...
git add .

echo.
echo [4/7] 检查将要提交的文件...
git status

echo.
echo [5/7] 提交文件...
git commit -m "Initial commit: Semantic Communication System with Adversarial Attacks

- Implemented semantic communication system for MNIST
- Added FGSM, PGD, and end-to-end adversarial attacks
- Complete experimental results and documentation
- All code tested and reproducible"

echo.
echo [6/7] 连接到远程仓库...
git remote remove origin 2>nul
git remote add origin https://github.com/cyrus28214/ZJU-course-Net-final.git

echo.
echo [7/7] 设置主分支并推送...
git branch -M main
git push -u origin main

echo.
echo ========================================
echo 上传完成！
echo ========================================
echo.
echo 如果遇到认证问题，请：
echo 1. 使用GitHub Personal Access Token代替密码
echo 2. 或配置SSH密钥
echo.
pause

