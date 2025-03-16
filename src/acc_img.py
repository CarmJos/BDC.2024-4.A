import numpy as np
from matplotlib import pyplot as plt


def main():
    epoch_list = []
    train_loss_list = []
    test_loss_list = []
    acc_list = []

    count = 0
    with open("../accuracy.csv", "r") as f:
        for line in f.readlines():
            count += 1
            data = line.strip().split(",")
            epoch = count
            test_loss = float(data[2])
            acc = float(data[3])
            epoch_list.append(epoch)
            test_loss_list.append(test_loss)
            acc_list.append(acc)

    plt.plot(epoch_list, test_loss_list, label="Test Loss")
    plt.plot(epoch_list, acc_list, label="Accuracy")

    plt.legend(["model"])
    plt.xticks(np.arange(0, 20, 1))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xlabel("Epoch")
    # plt.ylabel("test_loss(100%)")
    # plt.title("Test Loss")
    plt.show()


if __name__ == '__main__':
    main()
