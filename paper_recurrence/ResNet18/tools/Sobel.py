import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

kernalx = torch.Tensor([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
]).unsqueeze(0).unsqueeze(0)
kernaly = torch.Tensor([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
]).unsqueeze(0).unsqueeze(0)

def SobelConv(imgs_gray_tensor):
    gx = F.conv2d(input = imgs_gray_tensor, weight = kernalx, padding = 1)
    gy = F.conv2d(input = imgs_gray_tensor, weight = kernaly, padding = 1)
    g = torch.sqrt(gx**2+gy**2)
    plot(imgs_gray_tensor.squeeze(0), g.squeeze(0))
    return g

 
def plot(*imgs_tensor):
    unloader = transforms.ToPILImage()
    total = len(imgs_tensor)
    fig, ax = plt.subplots(total//3+1,3, figsize=(15, 5*(total//3+1)))
    for i in range(total):
        img = unloader(imgs_tensor[i])
        ax[i].imshow(img, cmap ='gray')
    plt.show()


