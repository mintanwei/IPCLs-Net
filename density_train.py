import os
import torch
import torch.nn as nn
import random
from torchvision import transforms
from mcnn_model import MCNN, VGGDN
from dataset import NBIDenseDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

dataset_root = "./NBI_dataset"
model_root = './density_models'
min_mae = 10000
min_epoch = 0
total_epoch = 1000

def get_transform(train):
    all_transforms = []
    # if train:
        # all_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
    all_transforms.append(transforms.ToTensor())
    return transforms.Compose(all_transforms)

if __name__ == "__main__":
    device = torch.device("cuda")
    # mcnn = MCNN().to(device)
    vggdn = VGGDN().to(device)
    criterion = nn.MSELoss(reduction="sum").to(device)
    # optimizer = torch.optim.SGD(mcnn.parameters(), lr=1e-6, momentum=0.95)
    optimizer = torch.optim.SGD(vggdn.parameters(), lr=1e-6, momentum=0.95)

    train_dataset = NBIDenseDataset(root=os.path.join(dataset_root, "train"), transforms=get_transform(train=True))
    test_dataset = NBIDenseDataset(root=os.path.join(dataset_root, "test"), transforms=get_transform(train=False))
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    if not os.path.exists(model_root):
        os.mkdir(model_root)

    train_loss_list = []
    epoch_list = []
    test_error_list = []
    for epoch in range(total_epoch):
        # mcnn.train()
        vggdn.train()
        epoch_loss = 0
        for i, (img, gt_dmap) in enumerate(train_dataloader):
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            gt_dmap = F.interpolate(gt_dmap, scale_factor=1/32)
            # forward propagation
            # et_dmap = mcnn(img)
            et_dmap = vggdn(img)
            # calculate loss
            # print(gt_dmap.size())
            # print(et_dmap.size())
            loss = criterion(et_dmap, gt_dmap)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print("epoch:",epoch,"loss:",epoch_loss/len(dataloader))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss / len(train_dataloader))
        # torch.save(mcnn.state_dict(), os.path.join(model_root, "%d.pt" % (epoch + 1)))
        torch.save(vggdn.state_dict(), os.path.join(model_root, "%d.pt" % (epoch + 1)))

        # mcnn.eval()
        vggdn.eval()
        mae = 0
        for i, (img, gt_dmap) in enumerate(test_dataloader):
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            # forward propagation
            # et_dmap = mcnn(img)
            et_dmap = vggdn(img)
            mae += abs(et_dmap.data.sum() - gt_dmap.data.sum()).item()
            del img, gt_dmap, et_dmap
        if mae / len(test_dataloader) < min_mae:
            min_mae = mae / len(test_dataloader)
            min_epoch = epoch
        test_error_list.append(mae / len(test_dataloader))
        print("Epoch:" + str(epoch) + " train_MSE:" + str(epoch_loss / len(train_dataloader)) + " test_MAE:" + str(mae / len(test_dataloader)) + " test_min_MAE:" + str(
            min_mae) + "@Epoch:" + str(min_epoch))

    print(train_loss_list)
    print(epoch_list)
    print(test_error_list)