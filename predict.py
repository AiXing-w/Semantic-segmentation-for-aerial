import os
from math import ceil
import torch
import torchvision
from torchvision import io
from utils.dataLoader import load_data_voc
from utils.dataConvert import loadColorMap
from utils.model import U_net, deeplabv3
import matplotlib.pyplot as plt


def label2image(pred, device):
    VOC_COLORMAP = loadColorMap()
    colormap = torch.tensor(VOC_COLORMAP, device=device)
    X = pred.long()
    return colormap[X, :]


def predict(net, device, img, test_iter):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(device)).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])


def read_voc_images(data_dir, is_train=True):
    images = []
    labels = []
    if is_train:
        with open(os.path.join(data_dir, 'train.txt')) as f:
            lst = [name.strip() for name in f.readlines()]

    else:
        with open(os.path.join(data_dir, 'test.txt')) as f:
            lst = [name.strip() for name in f.readlines()]

    for name in lst:
        image = io.read_image(os.path.join(data_dir, 'images', '{:03d}.jpg'.format(int(name))))
        label = io.read_image(os.path.join(data_dir, 'labels', '{:03d}.png'.format(int(name))))
        images.append(image)
        labels.append(label)

    return images, labels


def plotPredictAns(imgs):
    length = len(imgs)

    for i, img in enumerate(imgs):
        plt.subplot(ceil(length / 3), 3, i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.title("original images")

        if i == 1:
            plt.title("predict label")

        if i == 2:
            plt.title("true label")

    plt.show()

if __name__ == '__main__':
    voc_dir = './dataset/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_images, test_labels = read_voc_images(voc_dir, False)
    n = 4
    imgs = []
    batch_size = 2
    crop_size = (600, 600)  # 裁剪大小
    _, test_iter = load_data_voc(batch_size, crop_size, data_dir='dataset')
    model_choice = 'U-net'

    if model_choice == 'U-net':
        net = U_net()
    elif model_choice == 'deeplabv3+':
        net = deeplabv3(3, 6)

    if model_choice == 'U-net':
        model_path = os.path.join('model_weights', 'u-net-vgg16.pth')
    elif model_choice == 'deeplabv3+':
        model_path = os.path.join('model_weights', 'Semantic-deeplabv3.pth')
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    for i in range(n):
        crop_rect = (0, 0, 600, 600)
        X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
        pred = label2image(predict(net, device, X, test_iter), device)
        imgs += [X.permute(1, 2, 0), pred.cpu(),
                 torchvision.transforms.functional.crop(test_labels[i], *crop_rect).permute(1, 2, 0)]

    plotPredictAns(imgs)