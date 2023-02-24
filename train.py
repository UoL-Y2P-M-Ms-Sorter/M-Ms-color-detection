from torchvision import transforms, datasets
import json
import torch
import torch.nn as nn
import torch.optim as optim
from model import resnet18
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("{} is in use".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.CenterCrop([256, 256]),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.CenterCrop([256, 256]),
                                   transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    batch_size = 16

    train_dataset = datasets.ImageFolder("data/train", transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=8)

    val_dataset = datasets.ImageFolder("data/val", transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=8)

    net = resnet18(7, True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 50

    best_acc = 0.0
    train_loss = []
    val_acc = []
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader)) as pbar:
            for images, labels in train_loader:
                output = net(images.to(device))
                loss = loss_function(output, labels.to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.update(1)

        net.eval()
        running_acc = 0.0

        with torch.no_grad():
            with tqdm(total=len(val_loader)) as pbar:
                for images, labels in val_loader:
                    output = net(images.to(device))
                    predict = torch.max(output, dim=1)[1]

                    running_acc += torch.eq(predict, labels.to(device)).sum().item()

                    pbar.update(1)

        train_loss.append(running_loss / len(train_dataset))
        val_acc.append(running_acc / len(val_dataset))
        print('[Epoch %d] train_loss: %.3f val_acc: %.3f' %
              (epoch + 1, train_loss[-1], val_acc[-1]))

        if val_acc > best_acc:
            best_acc = val_acc

            torch.save(net.state_dict(), "weight/mmst.pth")

    x = np.arange(0, epochs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    loss = ax.plot(x, train_loss, '-', label='train_loss')
    ax2 = ax.twinx()
    acc = ax2.plot(x, val_acc, '-', label='val_acc')
    lns = loss + acc
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    ax.grid()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    plt.title("train_loss & val_acc per epoch")
    plt.savefig('loss_acc.pdf')
    plt.show()


if __name__ == '__main__':
    main()
