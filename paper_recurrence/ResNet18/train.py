#!/usr/othersoftware/anaconda3/bin/python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
from model import ResNet18 as Net

def train(lr:float=0.005, epoches:int=30, batch_size:int=256, pretained_path:str=None, out_path:str="models/resnet18.pth"):
    """
    LeNet model training method
    :param lr             learning rate, optional
    :param epoches        training epoches totally, optional
    :param batch_size     the size of mini batch, optional
    :param pretained_path the path of pretained weight of model, optional
    :param out_path       the path of the weight of model after training, optional 
    """
    data_train_loader = DataLoader(
        MNIST(
            root="../../datasets",
            download=True,
            transform=T.Compose([
                T.Resize((224,224)),
                T.RandomVerticalFlip(0.1),
                T.RandomHorizontalFlip(0.1),
                T.ToTensor()
            ])
        ), 
        batch_size=batch_size, 
        shuffle=True, ## 随机排列
        num_workers=8 ## 8进程
    )

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    model = Net(1).to(device)
    if (type(pretained_path)=='str'):
        model.load_state_dict(torch.load(pretained_path))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epoches):
        t = tqdm(enumerate(data_train_loader, 1))
        totalLoss = 0
        for index,(data, label) in t:
            data = data.to(device)
            label = label.to(device)
            predict = model(data)
            loss = criterion(predict, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss+=loss.item()
            accuracy = (predict.argmax(1)==label).sum().item()/label.shape[0]

            t.set_description("Epoch %d index %d"%(epoch, index))
            t.set_postfix(loss=totalLoss/index, accuracy=accuracy)

        torch.save(model.state_dict(), out_path)
    print("Training finished...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is ResNet18 training module')
    parser.add_argument('--lr','-l', default=0.005, type=float, help='learning rate, default is 0.005')
    parser.add_argument('--epoches', '-e', default=30, type=int, help='training epoches totally, default is 30')
    parser.add_argument('--batch_size', '-b', default=256, type=int, help='the size of mini batch, default is 256')
    parser.add_argument('--pretained_path', '-p', default=None, type=str, help='the path of pretained weight of model, default is None')
    parser.add_argument('--out_path', '-o', default="models/resnet18.pth", type=str, help='the path of the weight of model after training default is models/resnet18.pth')
    args = parser.parse_args()
    train(args.lr, args.epoches, args.batch_size, args.pretained_path, args.out_path)
