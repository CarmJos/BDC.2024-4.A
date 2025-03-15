import torchvision

import config
import dataset


def test_reg():
    matches = config.label_pattern.findall("沉17亮晶砂屑石灰岩2-5.png")
    if len(matches) > 1:
        extracted = matches[1]
        print(extracted)
    else:
        print("match failed")


def test_dataset():
    data_path = ["../data/source/"]
    ds = dataset.ImageClassifyDataset(data_path, False, [1, 2])
    for data in ds.data:
        print(f'{data["class"]} - {data["label"]} - {data["augment_index"]}')


def main():
    test_dataset()


if __name__ == '__main__':
    main()
