import os

# 你的文件夹路径
folder = './result_csv'

# 遍历文件夹内的文件
for filename in os.listdir(folder):
    if 'organism' in filename:
        # 替换 organism 为 WEO
        new_filename = filename.replace('organism', 'WEO')
        # 构造完整路径
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_filename)
        # 重命名
        os.rename(src, dst)
        print(f'Renamed: {filename} --> {new_filename}')
    else:
        print(f'Skipped: {filename}')
