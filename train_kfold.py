from engine import train_one_epoch, evaluate
import utils
import torch
from dataset import NBIFiveFoldDataset
from torchvision import transforms
from model import rcnn_model
from torch.utils.data import DataLoader
from visualize import visualize
import os


def get_transform(train):
    all_transforms = []
    all_transforms.append(transforms.ToTensor())
    return transforms.Compose(all_transforms)


def main(k, f):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = rcnn_model(k=k)
    model.to(device)

    fold = f
    five_fold_dataset = NBIFiveFoldDataset(get_transform(train=True))

    # define training and validation data loaders
    data_loader_train = DataLoader(five_fold_dataset[fold]["train"], batch_size=1, shuffle=True, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(five_fold_dataset[fold]["val"], batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    # construct an optimizer and a learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

    num_epochs = 50

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), os.path.join("./models", 'resnet_101_fold%d_epoch-%d-top%d.pt' % (fold, epoch + 1, k)))


if __name__ == '__main__':
    main(4, 0)
    