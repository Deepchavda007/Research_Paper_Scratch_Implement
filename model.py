from utils import nn, opt, F, models
from utils import torch
### second method of implement

class Alexnet(nn.Module):
    def __init__(self, num_classes):
        super(Alexnet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes),
            nn.Dropout(0.5),
            nn.Softmax(),
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x,-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
    

class Mobilenet(nn.Module):
    def __init__(self, num_classes):
        super(Mobilenet).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        self.avgpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax()
        )

    def mobilenet_block(self, x, in_ch, out_ch, stride):

        ## dor depwthwise convolution
        x = nn.DepthwiseConv2d(in_ch, out_ch, kernel_size = 3, padding = 'same', strides = stride)(x)
        x = nn.BatchNorm2d(in_ch)(x)
        x = nn.ReLU(inplace=True)(x)

        ## fr pointwise convolution
        x = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)(x)
        x = nn.BatchNorm2d(out_ch)(x)
        x = nn.ReLU(inplace=True)(x)

        return x

    def forward(self, x):
        # pass
        x = self.conv1(x)
        x = self.mobilenet_block(x, 32, 64, 1)
        x = self.mobilenet_block(x, 64, 128, 2)
        x = self.mobilenet_block(x, 128, 128, 1)
        x = self.mobilenet_block(x, 128, 256, 2)
        x = self.mobilenet_block(x, 256, 256, 1)
        x = self.mobilenet_block(x, 256, 512, 2)
        for _ in range(6):
            x = self.mobilenet_block(x, 512, 512, 1)

        x = self.mobilenet_block(x, 512, 1024, 2)
        x = self.mobilenet_block(x, 1024, 1024, 1)
        x = torch.flatten(x,-1)
        x = self.avgpool(x)
        x = self.fc1(x)

        return x
    



        
