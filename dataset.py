import os
import torch
from PIL import Image
from read_csv import csv_to_label_and_bbx
import numpy as np
from torch.utils.data import Subset, random_split, ConcatDataset


class NBIDataset(object):
    def __init__(self, root, transforms, nob3=False):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.boxes = csv_to_label_and_bbx(os.path.join(self.root, "annotations.csv"), nob3)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        annotations = self.boxes[self.imgs[idx]]
        boxes = annotations['bbx']
        labels = annotations['labels']

        # FloatTensor[N, 4]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Int64Tensor[N]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((labels.size()[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        # target["image_path"] = img_path
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
            # target = self.transforms(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class NBINewDataset(object):
    def __init__(self, root, transforms, train=True):
        self.root = root
        self.transforms = transforms
        if train:
            self.path = os.path.join(root, "train")
        else:
            self.path = os.path.join(root, "test")

        self.imgs = list(sorted(os.listdir(self.path)))

        self.boxes = csv_to_label_and_bbx(os.path.join(self.root, "annotations_all.csv"), img_names=self.imgs)


    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        annotations = self.boxes[self.imgs[idx]]
        boxes = annotations['bbx']
        labels = annotations['labels']

        # FloatTensor[N, 4]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Int64Tensor[N]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((labels.size()[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        # target["image_path"] = img_path
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
            # target = self.transforms(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class NBIFullDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.path = os.path.join(root, "all")
        self.imgs = list(sorted(os.listdir(self.path)))
        self.boxes = csv_to_label_and_bbx(os.path.join(self.root, "annotations.csv"), img_names=self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        annotations = self.boxes[self.imgs[idx]]
        boxes = annotations['bbx']
        labels = annotations['labels']

        # FloatTensor[N, 4]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Int64Tensor[N]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((labels.size()[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        # target["image_path"] = img_path
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
            # target = self.transforms(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class NBIDenseDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        density_path = os.path.join(self.root, "density_maps")
        density_map = np.load(os.path.join(density_path, self.imgs[idx][:-4] + ".npy"))
        density_map = torch.from_numpy(density_map)

        if self.transforms is not None:
            img = self.transforms(img)
            # target = self.transforms(target)

        return img, density_map

    def __len__(self):
        return len(self.imgs)


class NBIPatchDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = [x for x in list(sorted(os.listdir(root))) if x[-3:] == "png"]
        self.ans = np.load(os.path.join(root, "ans.npy"), allow_pickle=True).item()

    def __getitem__(self, idx):
        # img_path = os.path.join(self.root, "images", self.imgs[idx])
        # img = Image.open(img_path).convert("RGB")
        # density_path = os.path.join(self.root, "density_maps")
        # density_map = np.load(os.path.join(density_path, self.imgs[idx][:-4] + ".npy"))
        # density_map = torch.from_numpy(density_map)
        #
        # if self.transforms is not None:
        #     img = self.transforms(img)
        #     # target = self.transforms(target)

        return self.imgs[idx]

    def __len__(self):
        return len(self.imgs)


def split_index(K=5, len=100):
    idx = list(range(len))
    final_list = []
    for i in range(K):
        final_list.append(idx[(i*len)//K:((i+1)*len)//K])
    return final_list


def k_fold_index(K=5, len=100, fold=0):
    split = split_index(K, len)
    val = split[fold]
    train = []
    for i in range(K):
        if i != fold:
            train = train + split[i]
    return train, val


def stat_dataset(dataset):
    class_ids = {1: "A", 2: "B1", 3: "B2", 4: "B3"}
    stats = {"A": 0, "B1": 0, "B2": 0, "B3": 0}
    for img, target in dataset:
        for k in target['labels']:
            stats[class_ids[int(k)]] += 1
    print(stats)


def  NBIFiveFoldDataset(transforms):
    ds = NBIFullDataset(root="./NBI_full_dataset/", transforms=transforms)
    # n = len(ds)
    # for i in range(5):
    #     train_idx, val_idx = k_fold_index(5, n, i)
    #     train_subset = Subset(ds, train_idx)
    #     val_subset = Subset(ds, val_idx)
    #     print("Fold: %d" % i, len(train_subset), len(val_subset))
    #     stat_dataset(train_subset)
    #     stat_dataset(val_subset)
    torch.manual_seed(13)
    all_subsets = random_split(ds, [46, 46, 46, 45, 45])
    fold_i_subsets = []
    for i in range(5):
        val_subset = all_subsets[i]
        train_subset = ConcatDataset([all_subsets[j] for j in range(5) if j != i])
        fold_i_subsets.append({"train": train_subset, "val": val_subset})
        # print("Fold: %d" % i, len(train_subset), len(val_subset))
        # stat_dataset(train_subset)
        # stat_dataset(val_subset)
    return fold_i_subsets

if __name__ == '__main__':
    # ds = NBIFiveFoldDataset(None)
    di = "aaa".encode("UTF-8")
    result = eval(di)
    print(result)

