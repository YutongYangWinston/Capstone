#Author: Yutong Yang
#16/06/2022
#VGGnet
import torch

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.Conv = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, stride=2),

                                        torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, stride=2),

                                        torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, stride=2),

                                        torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2, stride=2))
#Fully connected layer
        self.Classes = torch.nn.Sequential(torch.nn.Linear(4 * 4 * 512, 1024),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(1024, 1024),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(1024, 3))


    def forward(self, input):
        x = self.Conv(input)
        x = x.view(-1, 4 * 4 * 512)
        x = self.Classes(x)
        return x
