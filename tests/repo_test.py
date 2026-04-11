import os

# 你的根目录
root_path = "./"  # 改成你要遍历的路径

# 定义三个空列表，用来收集所有结果
all_roots = []    # 所有目录
all_dirs = []     # 所有子文件夹
all_files = []    # 所有文件

# 遍历
for root, dirs, files in os.walk(root_path):
    # 把每一轮的 root 加入列表
    all_roots.append(root)
    
    # 把每一轮的 dirs 合并进大列表（用 extend 而不是 append）
    all_dirs.extend(dirs)
    
    # 把每一轮的 files 合并进大列表
    all_files.extend(files)

# 最终你就得到了三个完整的大列表
print("所有目录：", all_roots)
print("所有子文件夹：", all_dirs)
print("所有文件：", all_files)