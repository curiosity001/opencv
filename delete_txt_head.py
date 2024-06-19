import os

# 定义源文件夹和目标文件夹的路径
source_folder = r'D:\users\Desktop\Pedestrian dataset\增强后数据集\labels_txt_enhancement_personhead'
target_folder = r'D:\users\Desktop\Pedestrian dataset\增强后数据集\labels_txt_enhancement_person'

# 确保目标文件夹存在，如果不存在就创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的所有txt文件
for filename in os.listdir(source_folder):
    if filename.endswith(".txt"):
        source_file_path = os.path.join(source_folder, filename)
        target_file_path = os.path.join(target_folder, filename)

        # 打开源文件和目标文件
        with open(source_file_path, 'r') as source_file, open(target_file_path, 'w') as target_file:
            # 逐行读取源文件
            for line in source_file:
                # 检查每行的首个数字是否为0
                parts = line.strip().split()
                if parts and parts[0] == '0':
                    # 如果首个数字为0，将这一行写入目标文件
                    target_file.write(line)

# 完成后输出消息
print("处理完成！")
