#!/usr/othersoftware/anaconda3/bin/python
import torch
import torch.nn as nn

class ResNet18(nn.Module):
    class ResModule(nn.Module):
        def __init__(self, convModule1:nn.Module, convModule2:nn.Module, downsModule:nn.Module = None):
            super(ResNet18.ResModule, self).__init__()
            self.convModule = nn.Sequential(
                convModule1,
                convModule2
            )
            self.downsModule = downsModule
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x:torch.Tensor)->torch.Tensor:
            out1 = self.convModule(x)
            out2 = x if self.downsModule == None else self.downsModule(x)
            return self.relu(out1 + out2)



    def __init__(self, in_channels):
        super(ResNet18, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3, stride=2), # N*3*224*224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # N*64*56*56

            ## Lay1, contain two res module
            self.ResModule(
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1), # N*64*56*56
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1), # N*64*56*56
                    nn.BatchNorm2d(64)
                )
            ),
            self.ResModule(
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1), # N*64*56*56
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1), # N*64*56*56
                    nn.BatchNorm2d(64)
                )
            ),

            ## Lay 2 contain two res modules, first down sample, second not
            self.ResModule( # down sample res modules
                nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1, stride=2), # N*128*28*28
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(128, 128, 3, padding=1), # N*128*28*28
                    nn.BatchNorm2d(128)
                ),
                nn.Sequential(
                    nn.Conv2d(64, 128, 1, stride=2), # N*128*28*28, down sample
                    nn.BatchNorm2d(128) # 只有归一化，不同于前面还有ReLU
                )
            ),
            self.ResModule(
                nn.Sequential(
                    nn.Conv2d(128, 128, 3, padding=1), # N*128*28*28
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(128, 128, 3, padding=1), # N*128*28*28
                    nn.BatchNorm2d(128)
                )
            ),

                  ## Lay 3 contain two res modules, first down sample, second not
            self.ResModule( # down sample res modules
                nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1, stride=2), # N*256*14*14
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1), # N*256*14*14
                    nn.BatchNorm2d(256)
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, 1, stride=2), # N*256*14*14, down sample
                    nn.BatchNorm2d(256)
                )
            ),
            self.ResModule(
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1), # N*256*14*14
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1), # N*256*14*14
                    nn.BatchNorm2d(256)
                )
            ),

            ## Lay 4 contain two res modules, first down sample, second not
            self.ResModule( # down sample res modules
                nn.Sequential(
                    nn.Conv2d(256, 512, 3, padding=1, stride=2), # N*512*7*7
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(512, 512, 3, padding=1), # N*512*7*7
                    nn.BatchNorm2d(512)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 512, 1, stride=2), # N*512*7*7, down sample
                    nn.BatchNorm2d(512) # 只有归一化，不同于前面还有ReLU
                )
            ),
            self.ResModule(
                nn.Sequential(
                    nn.Conv2d(512, 512, 3, padding=1), # N*512*7*7
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(512, 512, 3, padding=1), # N*512*7*7
                    nn.BatchNorm2d(512)
                )
            ),

            nn.AvgPool2d(7)  ## N*512*1*1，全局平均池化为512维特征向量    
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 10),
            nn.Softmax(1)
        )

    
    def forward(self, x):
        return self.classifier(torch.flatten(self.feature(x), 1))

