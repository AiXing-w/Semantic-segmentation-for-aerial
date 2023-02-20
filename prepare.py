import os
import shutil
from random import shuffle
import torch
from torchvision import io

def semantic2dataset():
    # 航拍数据集转换成语义分割的数据集
    mark = 0
    path = 'Semantic segmentation dataset'
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    if not os.path.exists(os.path.join('dataset', 'images')):
        os.mkdir(os.path.join('dataset', 'images'))
    if not os.path.exists(os.path.join('dataset', 'labels')):
        os.mkdir(os.path.join('dataset', 'labels'))

    for i in range(1, 9):
        file = os.path.join(path, 'Tile {}'.format(i))
        images = os.path.join(file, 'images')
        masks = os.path.join(file, 'masks')
        for image, label in zip(os.listdir(images), os.listdir(masks)):
            shutil.copyfile(os.path.join(images, image), os.path.join('dataset', 'images', '{:03d}.jpg'.format(mark)))
            shutil.copyfile(os.path.join(masks, label), os.path.join('dataset', 'labels', '{:03d}.png'.format(mark)))
            mark += 1


def trainValSplit(path):
    length = len(os.listdir(os.path.join(path, 'images')))
    idx = [i for i in range(length)]
    shuffle(idx)
    with open(os.path.join(path, 'train.txt'), 'w') as f:
        for d in idx[:int(length * 0.8)]:
            f.write(str(d))
            f.write("\n")

    with open(os.path.join(path, 'test.txt'), 'w') as f:
        for d in idx[int(length * 0.8):]:
            f.write(str(d))
            f.write("\n")


def getMeanStd(path):
    length = len(os.listdir(path))
    means = torch.zeros(3)
    stds = torch.zeros(3)
    for name in os.listdir(path):
        img = io.read_image(os.path.join(path, name)).type(torch.float32) / 255
        for i in range(3):
            means[i] += img[i, :, :].mean()
            stds[i] += img[i, :, :].std()

    print("means:{}".format(means.div_(length)), "stds:{}".format(stds.div_(length)))


def writeColorClasses(color, classes):
    with open('color.txt', 'w') as f:
        for r, g, b in color:
            f.write('{}, {}, {}'.format(r, g, b))
            f.write('\n')

    with open('classes.txt', 'w') as f:
        for className in classes:
            f.write(className)
            f.write('\n')


if __name__ == '__main__':
    VOC_COLORMAP = [[226, 169, 41], [132, 41, 246], [110, 193, 228], [60, 16, 152], [254, 221, 58], [155, 155, 155]]
    VOC_CLASSES = ['Water', 'Land (unpaved area)', 'Road', 'Building', 'Vegetation', 'Unlabeled']
    writeColorClasses(VOC_COLORMAP, VOC_CLASSES)
    semantic2dataset()
    trainValSplit('./dataset')
    getMeanStd('./dataset/images')