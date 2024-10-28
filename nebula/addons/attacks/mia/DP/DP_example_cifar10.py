import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from opacus.data_loader import DPDataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
from torchmetrics import Accuracy, Precision


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "SampleConvNet"


def ML_train_test(train_set, test_set, size, seed):
    np.random.seed(seed)
    all_train_indices = np.arange(len(train_set))
    np.random.shuffle(all_train_indices)
    final_train_indices = all_train_indices[:size]
    final_train = Subset(train_set, final_train_indices)

    all_test_indices = np.arange(len(test_set))
    np.random.shuffle(all_test_indices)
    final_test_indices = all_test_indices[:int(size * 0.4)]
    final_test = Subset(test_set, final_test_indices)

    return final_train, final_test


def train(model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    precisions = []

    accuracy_metric = Accuracy(task='multiclass', num_classes=10).to(device)
    precision_metric = Precision(task='multiclass', average='macro', num_classes=10).to(device)

    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Update metrics
        acc = accuracy_metric(output, target)
        prec = precision_metric(output, target)
        accuracies.append(acc.item())
        precisions.append(prec.item())

    delta = 1e-5
    epsilon = privacy_engine.accountant.get_epsilon(delta)

    print(
        f"Train Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.6f} \t"
        f"Accuracy: {np.mean(accuracies):.6f} \t"
        f"Precision: {np.mean(precisions):.6f} \t"
        f"(ε = {epsilon:.2f}, δ = {delta})"
    )


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def save_model(model, epoch, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved at epoch {epoch} to {file_path}")


def main():
    device = torch.device("cuda")

    global_trainset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]
        ), )

    global_testset = datasets.CIFAR10(
        root="./data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]
        ),
    )

    train_set, test_set = ML_train_test(global_trainset, global_testset, 25000, 42)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=128,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = SampleConvNet().to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=1e-3,
                           betas=(0.851436, 0.999689),
                           amsgrad=True)
    privacy_engine = None

    privacy_engine = PrivacyEngine(secure_mode=False)
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1,
        max_grad_norm=1, )

    epochs_to_save = [10, 25, 50, 75, 100]
    for epoch in range(1, 101):
        train(model, device, train_loader, optimizer, privacy_engine, epoch)
        if epoch in epochs_to_save:
            save_model(model, epoch, f"model_epoch_12500_{epoch}.pth")

    test(model, device, test_loader)


if __name__ == '__main__':
    main()
