import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def calculate_histogram(image_folder):
    # 初始化累积的颜色通道直方图
    hist_r = np.zeros(256, dtype=np.float32)
    hist_g = np.zeros(256, dtype=np.float32)
    hist_b = np.zeros(256, dtype=np.float32)

    # 获取文件夹下所有图片文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

    count = 0
    # 读取每一张图片并计算颜色通道直方图
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        # 读取图片
        img = cv2.imread(image_path)

        if img is not None:
            # 转换为RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 计算每个颜色通道的直方图，并归一化
            hist_r += np.histogram(img_rgb[:, :, 0], bins=256, range=(0, 256))[0]
            hist_g += np.histogram(img_rgb[:, :, 1], bins=256, range=(0, 256))[0]
            hist_b += np.histogram(img_rgb[:, :, 2], bins=256, range=(0, 256))[0]

            # 释放图片以防内存溢出
            del img, img_rgb

            count += 1
            if count % 10 == 0:
                print(f"{count}/{len(image_files)}")

    # 计算总图片数
    num_images = len(image_files)
    hist_r /= num_images
    hist_g /= num_images
    hist_b /= num_images

    return hist_r, hist_g, hist_b


def plot_histogram(hist_r, hist_g, hist_b, output_path):
    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制直方图并填充颜色
    plt.fill_between(range(256), hist_r, color='red', alpha=0.5, label='Red Channel')
    plt.fill_between(range(256), hist_g, color='green', alpha=0.5, label='Green Channel')
    plt.fill_between(range(256), hist_b, color='blue', alpha=0.5, label='Blue Channel')

    # 添加标题和标签
    plt.title('Average RGB Color Channel Intensity Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # 添加图例
    plt.legend()

    # 保存图形
    plt.savefig(output_path)
    plt.close()


def main(image_folder, output_path):
    # 计算所有图片的颜色通道平均直方图
    hist_r, hist_g, hist_b = calculate_histogram(image_folder)

    # 绘制并保存最终的直方图
    plot_histogram(hist_r, hist_g, hist_b, output_path)


if __name__ == "__main__":
    # image_folder = "../data/source/南京大学教学沉积岩薄片照片数据集/"
    # image_folder = "../data/source/南京大学变质岩教学薄片照片数据集/"
    image_folder = "../data/source/南京大学火成岩教学薄片照片数据集/"
    output_path = "../color_stat.png"

    main(image_folder, output_path)
    print(f"Histogram saved to {output_path}")