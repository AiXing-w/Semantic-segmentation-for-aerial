import torch
from torch import nn
import os
from utils.dataLoader import load_data_voc
from utils.model import U_net, deeplabv3
from utils.losses import loss
from tqdm import tqdm


def train(net, epochs, train_iter, test_iter, device, loss, optimizer, model_path, auto_save):
    net = net.to(device)

    for epoch in range(epochs):
        net.train()

        train_loss = 0
        train_num = 0
        with tqdm(range(len(train_iter)), ncols=100, colour='red',
                  desc="train epoch {}/{}".format(epoch + 1, num_epochs)) as pbar:
            for i, (X, y) in enumerate(train_iter):
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                train_loss += l.detach()
                train_num += 1
                pbar.set_postfix({'loss': "{:.6f}".format(train_loss / train_num)})
                pbar.update(1)

        net.eval()
        test_loss = 0
        test_num = 0
        with tqdm(range(len(test_iter)), ncols=100, colour='blue',
                  desc="test epoch {}/{}".format(epoch + 1, num_epochs)) as pbar:
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                with torch.no_grad():
                    l = loss(y_hat, y)
                    test_loss += l.detach()
                    test_num += 1
                    pbar.set_postfix({'loss': "{:.6f}".format(test_loss / test_num)})
                    pbar.update(1)
        if (epoch + 1) % auto_save == 0:
            net.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    batch_size  = 2
    crop_size   = (500, 500) # 裁剪大小
    model_choice = 'deeplabv3+'  # 可选U-net、deeplabv3+
    in_channels = 3 # 输入图像通道
    out_channels = 6 # 输出标签类别
    num_epochs = 100 # 总轮次
    auto_save = 10 # 自动保存的间隔轮次
    lr = 2e-9 # 学习率

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_iter, test_iter = load_data_voc(batch_size, crop_size, data_dir='dataset')
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

    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train(net, num_epochs, train_iter, test_iter, device='cuda', loss=loss, optimizer=trainer, model_path=model_path, auto_save=auto_save)
    torch.save(net.state_dict(), model_path)
