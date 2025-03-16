import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import color


def calculate_texture_feature(image_folder):
    # 初始化累积的纹理相关系数
    texture_features = []

    # 获取文件夹下所有图片文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

    count = 0
    # 读取每一张图片并计算纹理相关系数
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        # 读取图片
        img = cv2.imread(image_path)

        if img is not None:
            # 转换为灰度图像
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 计算灰度共生矩阵（GLCM），这里使用默认的距离1和角度0（水平邻接）
            glcm = graycomatrix(img_gray, distances=[1], angles=[0], symmetric=True, normed=True)

            # 提取相关系数纹理特征
            correlation = graycoprops(glcm, prop='correlation')[0, 0]
            texture_features.append(correlation)

            # 释放图片以防内存溢出
            del img, img_gray, glcm

            count += 1
            if count % 10 == 0:
                print(f"{count}/{len(image_files)}")

    return texture_features


def plot_texture_histogram(texture_features, output_path):
    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制直方图
    plt.hist(texture_features, bins=30, color='gray', alpha=0.7, edgecolor='black')

    # 添加标题和标签
    plt.title('Average Texture Feature (Correlation) Histogram')
    plt.xlabel('Texture Correlation')
    plt.ylabel('Frequency')

    # 保存图形
    plt.savefig(output_path)
    plt.close()


def main(image_folder, output_path):
    # 计算所有图片的纹理相关系数
    texture_features = calculate_texture_feature(image_folder)

    # 绘制并保存纹理特征相关系数的直方图
    plot_texture_histogram(texture_features, output_path)


if __name__ == "__main__":
    image_folders = ["../data/source/南京大学教学沉积岩薄片照片数据集", "../data/source/南京大学变质岩教学薄片照片数据集", "../data/source/南京大学火成岩教学薄片照片数据集"]

    for image_folder in image_folders:
        output_path = f"../texture_stat{image_folder[image_folder.rindex('/') + 1:]}.png"
        main(image_folder, output_path)
        print(f"Texture histogram saved to {output_path}")