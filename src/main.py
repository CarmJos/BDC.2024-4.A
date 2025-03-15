from src import dataset, config
from src.trainer import Trainer


def main():
    # 数据集
    data_path = ["./data/source/南京大学变质岩教学薄片照片数据集/", "./data/source/南京大学火成岩教学薄片照片数据集", "./data/source/南京大学教学沉积岩薄片照片数据集"]
    train_dataset = dataset.ImageClassifyDataset(data_path, True, [1, 2])
    test_dataset = dataset.ImageClassifyDataset(data_path, True, [3])

    print(f"size of train sets: {len(train_dataset)}")
    print(f"size of test sets: {len(test_dataset)}")

    # 开始训练
    trainer = Trainer(train_dataset, test_dataset, config.num_epochs, config.learning_rate)
    trainer.start()
    # 保存模型
    trainer.save('../model.pth')


if __name__ == '__main__':
    main()
