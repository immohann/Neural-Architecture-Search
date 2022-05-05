import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


CIFAR10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def download_CIFAR10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
    torch.save(train_loader, "train_loader.pth")
    torch.save(test_loader, "test_loader.pth")


def load_CIFAR10():
    train_loader = torch.load("train_loader.pth")
    test_loader = torch.load("test_loader.pth")
    return train_loader, test_loader


def show_random(loader):
    # get a random image
    data_iter = iter(loader)
    image, label = data_iter.next()

    # show an image and its label
    img = torchvision.utils.make_grid(image[0]).numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    print(CIFAR10_classes[label[0]])


def plot_performance(epochs, title, train_performance, test_performance):
    x_axis = np.arange(epochs)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.plot(x_axis, train_performance)
    plt.plot(x_axis, test_performance)
    plt.legend(["Train", "Test"])
    plt.show()
