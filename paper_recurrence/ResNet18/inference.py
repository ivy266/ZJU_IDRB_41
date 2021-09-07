#!/usr/othersoftware/anaconda3/bin/python
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tools.Sobel import SobelConv
from tqdm.auto import tqdm
from PIL import Image
from model import ResNet18 as Net

def infer(model_path:str, batch_size:int = 1024):
    """
    model inference based on the MNIST datasets
    :param model_path  the path of the weight of models waited to inference
    :param batch_size  the size of mini batch
    """
    data_test_loader = DataLoader(
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

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = Net(1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad(): ## 关闭计算图，推理不用梯度
        t = tqdm(enumerate(data_test_loader, 1))
        aveLoss = 0
        aveAccuracy = 0
        for index,(data, label) in t:
            data = data.to(device)
            label = label.to(device)
            predict = model(data)
            loss = criterion(predict, label)
            aveLoss+=loss.item()
            accuracy = (predict.argmax(1)==label).sum().item()/label.shape[0]
            aveAccuracy += accuracy
            t.set_description("index %d"%(index))
            t.set_postfix(aveLoss=aveLoss/index, aveAccuracy=aveAccuracy/index)

def predict(model_path:str, img_path:str):
    """
    predict the handwriten number
    :param model_path  the path of the weight of models waited to inference
    :param img_path    the handwriten number figure's path
    """
    img = Image.open(img_path)
    loader = T.Compose([
        T.Resize((224,224)),
        T.ToTensor()
    ])
    img_Tensor = loader(img)
    img_gray_Tensor = (0.299 * img_Tensor[0] + 0.587 * img_Tensor[1] + 0.114 * img_Tensor[2]).unsqueeze(0)
    sobel_tensor = SobelConv(img_gray_Tensor.unsqueeze(0))

    model = Net(1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Predict value is ",model(sobel_tensor).argmax(1).item())

if __name__ == "__main__": ## only used for model inference
    parser = argparse.ArgumentParser(description='This is ResNet18 inference module')
    parser.add_argument('model_path', metavar='PATH', type=str, help='the path of the weight of models waited to inference')
    parser.add_argument('--batch_size', '-b', default=1024, type=int, help='the size of mini batch, default is 256')
    args = parser.parse_args()
    infer(args.model_path, args.batch_size)