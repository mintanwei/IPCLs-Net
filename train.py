from engine import train_one_epoch, evaluate
import utils
import torch
from dataset import NBINewDataset
from torchvision import transforms
from model import rcnn_model
from torch.utils.data import DataLoader, Subset
from visualize import visualize
import os


def get_transform(train):
    all_transforms = []
    # if train:
    # all_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
    # transforms.RandomHorizontalFlip
    all_transforms.append(transforms.ToTensor())
    return transforms.Compose(all_transforms)


def main(k):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = rcnn_model(k=k)
    model.to(device)

    # use our dataset and defined transformations
    root = "./NBI_new_dataset"

    dataset_train = NBINewDataset(root, get_transform(train=True), train=True)
    dataset_test = NBINewDataset(root, get_transform(train=False), train=False)

    # define training and validation data loaders
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    # construct an optimizer and a learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

    num_epochs = 50

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), os.path.join("./models", 'resnet34_epoch-%d-focal-adam-top%d.pt' % (epoch + 1, k)))

    # visualize(model, data_loader_test, device, generate_result=False)


if __name__ == '__main__':
    main(4)