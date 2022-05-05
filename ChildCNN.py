import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from Data import plot_performance
import cv2


# Traditional CNN Architecture #########################################################################################
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.fc = nn.Linear(900, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def collect_control_performance(epochs, num_classes, train_loader, test_loader):
    print(f"##################################\n"
          f"Traditional CNN evaluation...\n"
          f"##################################\n")
    # initialize child CNN
    child = CNN(num_classes).float()

    child_optimizer = optim.Adam(child.parameters(), lr=1e-4)
    child_scheduler = StepLR(child_optimizer, step_size=1, gamma=0.7)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        l, a = train_child(child, train_loader, child_optimizer, epoch)
        train_losses += [l]
        train_accuracies += [a]
        l, a = test_child(child, test_loader)
        test_losses += [l]
        test_accuracies += [a]
        child_scheduler.step()

    plot_performance(epochs, 'Control loss', train_losses, test_losses)
    plot_performance(epochs, 'Control accuracy', train_accuracies, test_accuracies)

    performance = sum(test_accuracies) / (epochs * 100)
    print(f"##################################\n"
          f"Control performance: {performance}\n"
          f"##################################\n")
    return performance



# DFA Child Architecture ###############################################################################################
class DFA_Child(nn.Module):
    def __init__(self, operations, num_classes):
        super(DFA_Child, self).__init__()
        self.operations = operations
        self.conv = nn.Conv2d(1, 4, 3)
        self.fc = nn.Linear(900, num_classes)

    def forward(self, x):
        x = x[0, 0].numpy()  # batch_size = 1 and channel = 1
        for operation in self.operations:
            x = operation(x)
        x = cv2.resize(x, (32, 32), interpolation=cv2.INTER_AREA)  # resize to standard
        x = torch.FloatTensor(x)[None, None, :]  # put it back in a channel and a batch
        x = self.conv(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def evaluate_child(epochs, selected_set, num_classes, train_loader, test_loader):
    print(f"##################################\n"
          f"Child's operations: {selected_set}\n"
          f"##################################\n")
    # initialize child CNN
    child = DFA_Child(selected_set, num_classes).float()

    child_optimizer = optim.Adam(child.parameters(), lr=1e-4)
    child_scheduler = StepLR(child_optimizer, step_size=1, gamma=0.7)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        l, a = train_child(child, train_loader, child_optimizer, epoch)
        train_losses += [l]
        train_accuracies += [a]
        l, a = test_child(child, test_loader)
        test_losses += [l]
        test_accuracies += [a]
        child_scheduler.step()

    plot_performance(epochs, 'Child loss', train_losses, test_losses)
    plot_performance(epochs, 'Child accuracy', train_accuracies, test_accuracies)

    performance = sum(test_accuracies) / (epochs * 100)
    print(f"##################################\n"
          f"Child's performance: {performance}\n"
          f"##################################\n")
    return performance


# train-test ###########################################################################################################
def train_child(model, train_loader, optimizer, epoch):
    model.train()
    N = len(train_loader)
    tot_loss = 0
    correct = 0

    for X, Y in train_loader:
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = torch.nn.CrossEntropyLoss()(Y_pred, Y)
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        correct += Y_pred[0].argmax().item() == Y[0].item()

    loss = tot_loss / N
    accuracy = 100.0 * correct / N
    print(f'Epoch {epoch}:\nTraining Loss: {loss}, Training Accuracy: {accuracy}%')
    return loss, accuracy


def test_child(model, test_loader):
    model.eval()
    N = len(test_loader)
    tot_loss = 0
    correct = 0

    with torch.no_grad():
        for X, Y in test_loader:
            Y_pred = model(X)

            tot_loss += torch.nn.CrossEntropyLoss()(Y_pred, Y).item()
            correct += Y_pred[0].argmax().item() == Y[0].item()

    loss = tot_loss / N
    accuracy = 100.0 * correct / N
    print(f'Testing Loss: {loss}, Testing Accuracy: {accuracy}%')
    return loss, accuracy

