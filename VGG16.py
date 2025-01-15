import torch
import torch.nn as nn

#VGG16 Architecture
VGG16 = ['64', '64', 'M', '128', '128', 'M', '256', '256', '256', 'M', '512', '512', '512', 'M', '512', '512', '512', 'M']
#Then flatten the layer and 4096 --> 4096 --> 1000 Linear layer

class VGG_net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self.develop_conv(VGG16)
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fcs(x)
        return x

    def develop_conv(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:

            if x != 'M':
                self.conv = nn.Conv2d(in_channels=int(in_channels), out_channels=int(x), kernel_size=(3,3), stride=(1,1), padding=(1,1))
                layers+=[self.conv,nn.BatchNorm2d(int(x)),nn.ReLU()]
                in_channels = x
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
                layers.append(self.maxpool)
        return nn.Sequential(*layers)

model=VGG_net(in_channels=3,num_classes=1000)
x = torch.randn(1,3,224,224)



