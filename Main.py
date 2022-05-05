import matplotlib.pyplot as plt

from Data import *
from ControllerRNN import *
from ChildCNN import *
from OperationSet import *
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch.distributions import Categorical

controller_hidden_dim = 512
controller_input_dim = 1
controller_num_layers = 1

child_num_layers = 2
child_train_epochs = 5

num_iterations = 100
num_classes = len(CIFAR10_classes)
num_operations = len(operation_set)


if __name__ == '__main__':
    # download_CIFAR10()
    train_loader, test_loader = load_CIFAR10()
    # show_random(train_loader)

    # collect control performance of a traditional CNN
    control_performance = collect_control_performance(child_train_epochs, num_classes, train_loader, test_loader)

    # initialize the controller RNN
    controller = Controller(controller_hidden_dim, controller_input_dim, controller_num_layers, num_operations).float()
    controller_optimizer = optim.Adam(controller.parameters(), lr=1e-4)
    controller_scheduler = StepLR(controller_optimizer, step_size=1, gamma=0.7)

    # main loop
    controller_train_losses = []
    rewards = []
    for iteration in range(num_iterations):
        controller.train()

        noise = np.array([[np.random.random(1)]])
        X = torch.from_numpy(noise).float()
        H = torch.zeros((controller_num_layers, 1, controller_hidden_dim))
        C = torch.zeros((controller_num_layers, 1, controller_hidden_dim))

        controller_optimizer.zero_grad()

        # let controller sample a number of operations
        selected_operations = []
        log_probs = []
        for i in range(child_num_layers):
            X, (H, C) = controller(X, (H, C))
            m = Categorical(X)
            operation = m.sample()
            O = operation.item()
            X = torch.FloatTensor([O])[None, None, :]
            log_probs += [m.log_prob(operation)]
            selected_operations += [O]

        reward = evaluate_child(child_train_epochs, translate(selected_operations), num_classes, train_loader, test_loader)
        rewards += [reward]


        loss = torch.cat([-log_prob * reward for log_prob in log_probs]).sum()
        loss.backward()
        controller_optimizer.step()

        loss = loss.item()
        print(f'Iteration {iteration}:\nController\'s Loss: {loss}')
        controller_train_losses += [loss]
        controller_scheduler.step()

    torch.save(controller.state_dict(), "controllerRNN.pt")

    x_axis = np.arange(num_iterations)
    plt.title("Controller Loss vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(x_axis, controller_train_losses)
    plt.show()

    plt.title("Controller Performance vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Performance")
    plt.plot(x_axis, rewards)
    plt.show()


