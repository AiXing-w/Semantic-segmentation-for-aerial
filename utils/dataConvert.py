import torch
import torchvision

def loadColorMap():
    lst = []
    with open('color.txt') as f:
        for line in f.readlines():
            line = line.strip()
            r, g, b = line.split(',')
            lst.append([int(r), int(g), int(b)])
    return lst


def voc_colormap2label():
    # RGB标签转换成数值标签的映像
    VOC_COLORMAP = loadColorMap()
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]
            ] = i

    return colormap2label


def voc_rand_crop(feature, label, height, width):
    # 数据裁剪
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width)
    )
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label


def voc_label_indices(colormap, colormap2label):
    # RGB标签转到数值标签
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


def one_hot(target, num_classes=6, device='cuda'):
    b, h, w = target.size()
    hot = torch.zeros((num_classes, b, h, w)).to(device)
    for i in range(num_classes):
        idx = (target == i)
        hot[i, idx] = 1.0

    return hot.permute((1, 2, 3, 0))