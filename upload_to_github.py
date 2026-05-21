# upload_to_github.py - 上传项目到 GitHub
import os
import shutil

print("=" * 70)
print("准备上传项目到 GitHub")
print("=" * 70)

# 1. 删除 .git 目录
if os.path.exists(".git"):
    print("\n[1] 删除旧的 .git 目录...")
    shutil.rmtree(".git")
    print("  完成")

# 2. 初始化 Git
print("\n[2] 初始化 Git 仓库...")
os.system("git init")
print("  完成")

# 3. 添加所有文件
print("\n[3] 添加文件...")
os.system('git add .')
print("  完成")

# 4. 检查状态
print("\n[4] 检查文件状态...")
os.system("git status")

print("\n" + "=" * 70)
print("准备完成!")
print("=" * 70)

print("\n接下来，请执行以下命令：")
print()
print("1. 设置你的 Git 用户信息（如果还没有设置）：")
print("   git config user.name 'Your Name'")
print("   git config user.email 'your@email.com'")
print()
print("2. 创建初始提交：")
print("   git commit -m 'Initial commit'")
print()
print("3. 添加远程仓库（将 URL 替换为你的仓库地址）：")
print("   git remote add origin https://github.com/quewww/ceramic-restoration.git")
print()
print("4. 推送到 GitHub：")
print("   git branch -M main")
print("   git push -u origin main")
print()
print("=" * 70)
