from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import os
data_dir = 'data'

# def CNN model
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


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    batch_size = 32

    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ])

    # train
    train_dataset = datasets.ImageFolder(root="./data/train", transform=transform)
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size)
    # test
    test_dataset = datasets.ImageFolder(root="./data/val", transform=transform)

    test_loader = DataLoader(test_dataset,
                             shuffle=True,
                             batch_size=batch_size)
    if os.path.exists("model.pth"):
        model = torch.load("model.pth").to(device)
    else:
        model = Net().to(device)

    print(model)
    # loss
    criterion = nn.CrossEntropyLoss()
    # opt
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_list = []
    train_acc = []
    test_acc = []
    for epoch in range(40):
        print(f'epoch：{epoch}')
        for batch, (X, y) in enumerate(train_loader):
            y_pred = model(X.to(device))
            #
            loss = criterion(y_pred, y.to(device))
            #
            optimizer.zero_grad()
            #
            loss.backward()
            #
            optimizer.step()
            #
        loss_list.append(loss.data.item())
        print("train loss------------", loss.data.item())
        # cal train accuracy

        # cal test accuracy
        rets = []
        total = 0
        train_total = 0
        correct = 0
        train_correct = 0
        with torch.no_grad():
            for data in train_loader:
                X, y = data
                y.to(device)
                y_pred = model(X.to(device))
                # max_val and  index
                _, predicted = torch.max(y_pred.data, dim=1)
                #predicted.to(device)
                train_total += y.size(0)
                train_correct += (predicted.to('cpu') == y).sum().item()
        train_acc.append(train_correct/train_total)
        print('accuracy on train set: %.2f %% ' % (100.0 * (train_correct / train_total)))
        with torch.no_grad():
            for data in test_loader:
                X, y = data
                y.to(device)
                y_pred = model(X.to(device))
                # max_val and  index
                _, predicted = torch.max(y_pred.data, dim=1)
                #predicted.to(device)
                total += y.size(0)
                correct += (predicted.to('cpu') == y).sum().item()
        test_acc.append(correct / total)
        print('accuracy on test set: %.2f %% ' % (100.0 * (correct / total)))

    print(train_acc)
    print(test_acc)

    # save the model
    model.to('cpu')
    torch.save(model, 'model.pth')
    # acc plot
    acc_train = train_acc
    acc_test = test_acc
    plt.figure(dpi=200)
    plt.plot(range(40), acc_train, label="train accuracy", marker='o')
    # plt.scatter(range(40),acc_train,label="train accuracy")
    plt.plot(range(40), acc_test, label="val accuracy", marker='o')
    # plt.scatter(range(40),acc_train,label="train accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
