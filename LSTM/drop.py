import os

folder = r"D:\桌面\skeleton+D0-30000 - 副本"

reserved = ["A023", "A043", "A050"]

for filename in os.listdir(folder):
    name, ext = os.path.splitext(filename)
    if ext == ".skeleton" and name.endswith(tuple(reserved)):
        # 保留指定的文件
        continue
    else:
        # 删除其他文件
        file_path = os.path.join(folder, filename)
        os.remove(file_path)

print("Done")