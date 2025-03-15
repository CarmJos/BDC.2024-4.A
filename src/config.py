# 通过文件名识别标注的正则表达式
# 例: 沉17亮晶砂屑石灰岩2-5.png
# matches[0]: 大类，如: 沉
# matches[1]: 子类，如: 亮晶砂屑石灰岩
label_pattern = r'[\u4e00-\u9fff]+'
# 用来识别每个子类的编号，以区分训练集和测试集
# 如 matches[0]: 1-1
id_pattern = r"\d-\d"


# 训练参数
batch_size = 64
learning_rate = 0.01
num_epochs = 20

# 数据增强
num_augmented_samples = 5
