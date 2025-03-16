import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern


def calculate_lbp_histogram(image_folder, radius=1, n_points=8):
    # 初始化累积的LBP直方图
    lbp_histograms = []

    # 获取文件夹下所有图片文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

    count = 0
    # 读取每一张图片并计算LBP直方图
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        # 读取图片
        img = cv2.imread(image_path)

        if img is not None:
            # 转换为灰度图像
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 计算LBP特征，使用skimage.feature.local_binary_pattern
            lbp = local_binary_pattern(img_gray, n_points, radius, method="uniform")

            # 计算LBP直方图
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            lbp_hist = lbp_hist.astype('float')

            # 归一化直方图
            lbp_hist /= lbp_hist.sum()

            lbp_histograms.append(lbp_hist)

            # 释放图片以防内存溢出
            del img, img_gray, lbp

            count += 1
            if count % 10 == 0:
                print(f"{count}/{len(image_files)}")

    return lbp_histograms


def plot_lbp_histogram(lbp_histograms, output_path):
    # 计算所有图像的平均LBP直方图
    avg_lbp_hist = np.mean(lbp_histograms, axis=0)

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制直方图
    plt.bar(range(len(avg_lbp_hist)), avg_lbp_hist, width=0.8, color='gray', edgecolor='black')

    # 添加标题和标签
    plt.title('Average LBP Histogram')
    plt.xlabel('LBP Pattern')
    plt.ylabel('Frequency')

    # 保存图形
    plt.savefig(output_path)
    plt.close()


def main(image_folder, output_path):
    # 计算所有图片的LBP直方图
    lbp_histograms = calculate_lbp_histogram(image_folder)

    # 绘制并保存平均LBP直方图
    plot_lbp_histogram(lbp_histograms, output_path)


if __name__ == "__main__":
    image_folders = ["../../data/source/南京大学教学沉积岩薄片照片数据集",
                     "../../data/source/南京大学变质岩教学薄片照片数据集",
                     "../../data/source/南京大学火成岩教学薄片照片数据集"]

    for image_folder in image_folders:
        output_path = f"../../lbp_stat{image_folder[image_folder.rindex('/') + 1:]}.png"
        main(image_folder, output_path)
        print(f"Texture LBP histogram saved to {output_path}")
