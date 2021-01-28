import os
import torch
import torch.nn as nn
import random
from torchvision import transforms, datasets
from mcnn_model import MCNN, VGGDN
from dataset import NBIDenseDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from sklearn import metrics
from skimage import io
import matplotlib.pyplot as plt

dataset_root = "./NBI_patch_dataset"
model_root = './patch_models'
min_epoch = 0
total_epoch = 100

def eval_cls_metrics(gt_label, pred, class_names):
    # plot confusion matrix
    # conf_matrix = metrics.confusion_matrix(gt_label, pred)
    # conf_matrix_disp = metrics.ConfusionMatrixDisplay(conf_matrix, class_names)
    # conf_matrix_disp.plot(cmap=plt.cm.Blues)
    # plt.pause(0.001)

    # print classification report
    print(metrics.classification_report(gt_label, pred, target_names=class_names, output_dict=False))

def get_transform(train):
    all_transforms = []
    # if train:
        # all_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
    all_transforms.append(transforms.Resize((32, 32)))
    all_transforms.append(transforms.ToTensor())
    return transforms.Compose(all_transforms)

if __name__ == "__main__":
    device = torch.device("cuda")
    model = torchvision.models.vgg16(pretrained=False, num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, "train_full"), transform=get_transform(train=True))
    test_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, "test"), transform=get_transform(train=False))
    train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=False)

    if not os.path.exists(model_root):
        os.mkdir(model_root)

    train_loss_list = []
    test_loss_list = []
    epoch_list = []

    for epoch in range(total_epoch):
        model.train()
        train_loss = 0
        for i, (img, label) in enumerate(train_dataloader):
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            loss = criterion(output, label)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_list.append(epoch)
        train_loss_list.append(train_loss / len(train_dataloader))
        torch.save(model.state_dict(), os.path.join(model_root, "%d.pt" % (epoch + 1)))

        test_loss = 0

        model.eval()
        gt_label = []
        pred = []
        for i, (img, label) in enumerate(test_dataloader):
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            gt_label.append(label.view(-1).cpu())
            _, pred_label = output.max(1)
            pred.append(pred_label.view(-1).cpu())
            loss = criterion(output, label)
            test_loss += loss.item()
        test_loss_list.append(test_loss / len(test_dataloader))

        print("Epoch:" + str(epoch) + " train_loss:" + str(train_loss / len(train_dataloader)) + " test_loss:" +
              str(test_loss / len(test_dataloader)))

        gt_label = torch.cat(gt_label).numpy()
        pred = torch.cat(pred).detach().numpy()
        eval_cls_metrics(gt_label, pred, test_dataset.classes)
