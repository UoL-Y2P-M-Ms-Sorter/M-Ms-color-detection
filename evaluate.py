import torch
from sklearn.metrics import accuracy_score
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns, pandas as pd
import matplotlib.pyplot as plt
from model import resnet18
import json




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("{} is in use".format(device))

    net = resnet18(num_classes=7).to(device)
    net.load_state_dict(torch.load("weight/mms.pth", map_location=device))
    net.eval()

    with open('./class_indices.json', "r") as f:
        class_indict = json.load(f)

    class_correct = [0.] * 7
    class_total = [0.] * 7
    y_test, y_pred = [], []
    X_test = []

    BATCH_SIZE = 16
    data_transform = transforms.Compose([transforms.CenterCrop([256, 256]),
                                         transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_dataset = datasets.ImageFolder("data/val", transform=data_transform)  # 测试集数据
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                 num_workers=8)  # 加载数据


    classes = list(class_indict.values())

    with torch.no_grad():
        for images, labels in val_loader:
            X_test.extend([_ for _ in images])

            outputs = net(images.to(device))
            predict = torch.max(outputs, dim=1)[1]
            c = torch.eq(predict, labels.to(device))

            for i, label in enumerate(labels):
                class_correct[label] += c[i].item()
                class_total[label] += 1
            y_pred.extend(predict.numpy())
            y_test.extend(labels.cpu().numpy())

    for i in range(7):
        print(f"Accuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:2.0f}%")


    ac = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=classes)
    print("Accuracy is :", ac)
    print(cr)


    labels = pd.DataFrame(cm).applymap(lambda v: f"{v}" if v != 0 else f"")
    plt.figure()
    sns.heatmap(cm, annot=labels, fmt='s', xticklabels=classes, yticklabels=classes, linewidths=0.1)
    plt.savefig("1.pdf")
    plt.show()
