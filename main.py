from engine import train_one_epoch, evaluate
import utils
import torch
from dataset import NBIDataset
from torchvision import transforms
from model import rcnn_model
from torch.utils.data import DataLoader, Subset
from demo import visualize
import os


def get_transform(train):
    all_transforms = []
    # if train:
    #     all_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
    all_transforms.append(transforms.ToTensor())
    # if train:
    #     all_transforms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(all_transforms)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = rcnn_model
    model.to(device)

    # use our dataset and defined transformations
    root = "./NBI_dataset"

    dataset_train = NBIDataset(os.path.join(root, "train"), get_transform(train=True), nob3)
    dataset_test = NBIDataset(os.path.join(root, "test"), get_transform(train=False), nob3)

    print(nob3)

    # split the dataset in train and test set (total: 100)
    # torch.manual_seed(0)
    # indices = torch.randperm(len(dataset_train)).tolist()
    # print(indices)
    # dataset_train = Subset(dataset_train, list(range(len(dataset_train)))[:-20])
    # dataset_test = Subset(dataset_test, list(range(len(dataset_test)))[-20:])

    # define training and validation data loaders
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    # construct an optimizer and a learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

    num_epochs = 40

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), os.path.join("./models", 'model_epoch-%d-ce.pt' % (epoch + 1)))

    visualize(model, data_loader_test, device)



if __name__ == '__main__':
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(device)
    # model = rcnn_model
    # root = "./NBI_dataset"
    # dataset = NBIDataset(root, get_transform(train=True))
    # data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)
    # images, targets = next(iter(data_loader))
    # images = list(image for image in images)
    # targets = [{k: v for k, v in t.items()} for t in targets]
    # print(images)
    # print(targets)
    #
    # output = model(images, targets)   # Returns losses and detections
    # print(output)

    # For inference
    # model.eval()
    # x = [torch.rand(3, 100, 100), torch.rand(3, 100, 100)]
    # predictions = model(x)           # Returns predictions
    # print(predictions)

    main()