import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import GradConv2d,GradLinear
# from torchsummary import summary
# from ptflops import get_model_complexity_info
'''
modified to fit dataset size
'''
NUM_CLASSES = 10


class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, zero_grad_mea=False):
        self.name = "AlexNet"
        super(AlexNet, self).__init__()
        if zero_grad_mea == True:
            Conv2d = GradConv2d
            Linear = GradLinear
        else:
            Conv2d = nn.Conv2d
            Linear = nn.Linear
        self.features = nn.Sequential(
            Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            Linear(4096, 4096),
            nn.ReLU(inplace=True),
            Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

def test():
    net = AlexNet()
    print(net.name)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
    summary(net, input_size=(3, 32, 32), device='cpu')
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True, print_per_layer_stat=True,
                                                 verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
if __name__ == '__main__':
    test()