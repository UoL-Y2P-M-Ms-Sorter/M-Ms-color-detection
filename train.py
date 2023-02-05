from torchvision import transforms, datasets
import json
import torch
import torch.nn as nn
import torch.optim as optim
from model import resnet18
from tqdm import tqdm

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("{} is in use".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop([224, 224]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([256, 256]),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    batch_size = 16

    train_dataset = datasets.ImageFolder("data/train", transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=8)

    color_list = train_dataset.class_to_idx
    color_dict = dict((val, key) for key, val in color_list.items())
    json_str = json.dumps(color_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    val_dataset = datasets.ImageFolder("data/val", transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=8)

    net = resnet18(6, True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 5

    best_acc = 0.0
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

        train_loss = running_loss / len(train_dataset)
        val_acc = running_acc / len(val_dataset)
        print('[Epoch %d] train_loss: %.3f val_acc: %.3f' %
              (epoch + 1, train_loss, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc

            torch.save(net.state_dict(), "weight/mms.pth")


if __name__ == '__main__':
    main()
