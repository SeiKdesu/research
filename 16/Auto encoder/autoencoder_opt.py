import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib import rc
from autoencoder_rbf import *

BATCH_SIZE = 100

# データの準備
trainval_data = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
train_size = int(len(trainval_data) * 0.8)
val_size = int(len(trainval_data) * 0.2)
train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

images, labels = next(iter(train_loader))
print("images_size:", images.size())   # images_size: torch.Size([100, 1, 28, 28])
print("label:", labels[:10])   # label: tensor([7, 6, 0, 6, 4, 8, 5, 2, 2, 3])

image_numpy = images.detach().numpy().copy()
plt.imshow(image_numpy[0, 0, :, :], cmap='gray')

# Encoderのクラス
class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.lr = nn.Linear(7, 300)
        self.lr2 = nn.Linear(300, 100)
        self.lr_ave = nn.Linear(100, z_dim)   # average
        self.lr_dev = nn.Linear(100, z_dim)   # log(sigma^2)
        self.relu = nn.ReLU()
  
    def forward(self, x):
        x = self.lr(x)
        x = self.relu(x)
        x = self.lr2(x)
        x = self.relu(x)
        ave = self.lr_ave(x)    # average
        log_dev = self.lr_dev(x)    # log(sigma^2)
        ep = torch.randn_like(ave)   # 平均0分散1の正規分布に従うz_dim次元の乱数
        z = ave + torch.exp(log_dev / 2) * ep   # 再パラメータ化トリック
        return z, ave, log_dev

# Decoderのクラス
class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.lr = nn.Linear(z_dim, 100)
        self.lr2 = nn.Linear(100, 300)
        self.lr3 = nn.Linear(300, 28*28)
        self.relu = nn.ReLU()
  
    def forward(self, z):
        x = self.lr(z)
        x = self.relu(x)
        x = self.lr2(x)
        x = self.relu(x)
        x = self.lr3(x)
        x = torch.sigmoid(x)   # MNISTのピクセル値の分布はベルヌーイ分布に近いためシグモイド関数を適用
        return x

# VAEのクラス
class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
  
    def forward(self, x):
        z, ave, log_dev = self.encoder(x)
        x = self.decoder(z)
        return x, z, ave, log_dev

def criterion(predict, target, ave, log_dev):
    # target も predict に合わせて 784 次元に変形
    target = target.view(-1, 28*28)
    bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
    loss = bce_loss + kl_loss
    return loss


# ハイパーパラメータの設定
z_dim = 2
num_epochs = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VAE(z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)

history = {"train_loss": [], "val_loss": [], "ave": [], "log_dev": [], "z": [], "labels": []}

# 学習ループ
for epoch in range(num_epochs):
    model.train()
    for i, (x, labels) in enumerate(zip(train_data,label_data)):
        input = x.view(-1, 28*28).to(device).float()  # 入力を (batch_size, 28*28) に変形
        output, z, ave, log_dev = model(input)

        
        history["ave"].append(ave)
        history["log_dev"].append(log_dev)
        history["z"].append(z)
        history["labels"].append(labels)
        
        loss = criterion(output, input, ave, log_dev)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 50 == 0:
            print(f'Epoch: {epoch+1}, loss: {loss: 0.4f}')
        history["train_loss"].append(loss.item())

    # 検証ループ
    model.eval()
    with torch.no_grad():
        for i, (x, labels) in enumerate(val_loader):
            input = x.view(-1, 7).to(device).float()
            output, z, ave, log_dev = model(input)
            loss = criterion(output, input, ave, log_dev)
            history["val_loss"].append(loss.item())
        
        print(f'Epoch: {epoch+1}, val_loss: {loss: 0.4f}')
    
    scheduler.step()

# 損失のプロット
train_loss_np = np.array(history["train_loss"])
plt.plot(train_loss_np)
val_loss_np = np.array(history["val_loss"])
plt.plot(val_loss_np)

# zのプロット
z_tensor = torch.stack(history["z"])
labels_tensor = torch.stack(history["labels"])
z_np = z_tensor.cpu().detach().numpy()
labels_np = labels_tensor.cpu().detach().numpy()

batch_num = 10
plt.figure(figsize=[10,10])
for label in range(10):
    x = z_np[:batch_num, :, 0][labels_np[:batch_num, :] == label]
    y = z_np[:batch_num, :, 1][labels_np[:batch_num, :] == label]
    plt.scatter(x, y, label=label, s=15)
    plt.annotate(label, xy=(np.mean(x), np.mean(y)), size=20, color="black")
plt.legend(loc="upper left")

model.to("cpu")
