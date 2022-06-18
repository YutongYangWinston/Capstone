#Test file
import torch
from PIL import Image
from torchvision import transforms
from torchvision import datasets
from train_model import Net
import numpy as np


def get_predict(filename):
    model = torch.load('model.pth')
    transform = transforms.Compose([
        transforms.Resize([64, 64]),  # Scale 64×64   Resize
        transforms.ToTensor()
    ])
    img = Image.open(f"./upload/{filename}").convert('RGB')
    img = transform(img)
    img = img.view((-1, 3, 64, 64))
    predict = model(img)
    class_index = np.argmax(predict.detach().numpy())
    train_dataset = datasets.ImageFolder(root="./data/train", transform=transform)
    return train_dataset.classes[class_index]


if __name__ == '__main__':
    model = torch.load('model.pth')
    transform = transforms.Compose([
        transforms.Resize([64, 64]),  # Scale 64×64
        transforms.ToTensor()
    ])

    img = Image.open("./data/train/C2_02_003.png").convert('RGB')
    img = transform(img)
    img = img.view((-1, 3, 64, 64))
    predict = model(img)
    class_index = np.argmax(predict.detach().numpy())
    train_dataset = datasets.ImageFolder(root="./data/train", transform=transform)
    print(train_dataset.classes[class_index])
