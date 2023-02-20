import torch
import torchvision
import os
from utils.dataConvert import voc_colormap2label, voc_rand_crop, voc_label_indices


class SemanticDataset(torch.utils.data.Dataset):
    # 加载航拍数据集
    def __init__(self, is_train, crop_size, data_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.4813, 0.4844, 0.4919], std=[0.2467, 0.2478, 0.2542])
        self.crop_size = crop_size
        self.data_dir = data_dir
        self.is_train = is_train
        self.colormap2label = voc_colormap2label()

        txt_fname = os.path.join(data_dir, 'train.txt' if self.is_train else 'test.txt')

        with open(txt_fname, 'r') as f:
            self.images = f.read().split()

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def pad_params(self, crop_h, crop_w, img_h, img_w):
        hight = max(crop_h, img_h)
        width = max(crop_w, img_w)
        y_s = (hight - img_h) // 2
        x_s = (width - img_w) // 2
        return hight, width, y_s, x_s

    def pad_image(self, hight, width, y_s, x_s, feature):
        zeros = torch.zeros((feature.shape[0], hight, width))
        zeros[:, y_s:y_s + feature.shape[1], x_s:x_s + feature.shape[2]] = feature
        return zeros

    def __getitem__(self, idx):
        mode = torchvision.io.image.ImageReadMode.RGB

        feature = torchvision.io.read_image(os.path.join(
            self.data_dir, 'images', '{:03d}.jpg'.format(int(self.images[idx]))))
        label = torchvision.io.read_image(os.path.join(
            self.data_dir, 'labels', '{:03d}.png'.format(int(self.images[idx]))), mode)

        c_h, c_w, f_h, f_w = self.crop_size[0], self.crop_size[1], feature.shape[1], feature.shape[2]
        if f_h < c_h or f_w < c_w:
            higth, width, y_s, x_s = self.pad_params(c_h, c_w, f_h, f_w)
            feature = self.pad_image(higth, width, y_s, x_s, feature)
            label = self.pad_image(higth, width, y_s, x_s, label)

        feature = self.normalize_image(feature)

        feature, label = voc_rand_crop(feature, label,
                                       *self.crop_size)
        label = voc_label_indices(label, self.colormap2label)
        return (feature, label)

    def __len__(self):
        return len(self.images)


def load_data_voc(batch_size, crop_size, data_dir='./dataset'):
    # 批量加载航拍数据集
    train_iter = torch.utils.data.DataLoader(SemanticDataset(True, crop_size, data_dir), batch_size, shuffle=True,
                                             drop_last=True)
    test_iter = torch.utils.data.DataLoader(SemanticDataset(False, crop_size, data_dir), batch_size, shuffle=False,
                                            drop_last=True)

    return train_iter, test_iter
