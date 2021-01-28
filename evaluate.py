from visualize import visualize
import os
import torch
from main import get_transform
from dataset import NBINewDataset
from torch.utils.data import DataLoader
import utils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from model import rcnn_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 5

# use our dataset and defined transformations
root = "./NBI_new_dataset"
# dataset_train = NBINewDataset(root, get_transform(train=True), train=True)
dataset_test = NBINewDataset(root, get_transform(train=False), train=False)

# define training and validation data loaders
# data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, collate_fn=utils.collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

# box_score_threshs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# box_nms_threshs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
box_score_threshs = [0.3]
box_nms_threshs = [0.3]

# for k in range(1, 12):
for k in [4]:
    for bst in box_score_threshs:
        for bnt in box_nms_threshs:
            results = {}
            f = open('./experiments/resnet101_concat_top%d_bst_%.2f_bnt_%.2f.txt' % (k, bst, bnt), 'w')
            for epoch in range(6, 51, 2):
            # for epoch in [20]:
                model = rcnn_model(box_score_thresh=bst, box_nms_thresh=bnt, knn_features=True, k=k)
                # model = rcnn_model(box_score_thresh=bst, box_nms_thresh=bnt, knn_features=False)

                model.to(device)

                # model_path = os.path.join("./models", 'model_epoch-%d-focal-adam-top%d.pt' % (epoch, k))
                model_path = os.path.join("./models", 'resnet101_epoch-%d-focal-adam-top%d.pt' % (epoch, k))
                net_state_dict = torch.load(model_path)
                model.load_state_dict(net_state_dict)

                print("Epoch:%d\n" % epoch)
                f.write("Epoch:%d\n" % epoch)
                print_str, matrix = visualize(model, data_loader_test, device, generate_result=False)
                f.write(print_str)
                f.write("\n")
                results[epoch] = matrix[4][3:7]

            maxf2 = 0
            maxf2epoch = 0
            for epoch, v in results.items():
                f.write("Epoch:%d, Sensitivity=%.3f, Precision=%.3f, Recall=%.3f, F2=%.3f\n" % (epoch, v[0], v[1], v[2], v[3]))
                if v[3] > maxf2:
                    maxf2 = v[3]
                    maxf2epoch = epoch

            if maxf2 > 0:
                v = results[maxf2epoch]
                f.write("\nBest Epoch:%d, Sensitivity=%.3f, Precision=%.3f, Recall=%.3f, F2=%.3f\n" % (
                maxf2epoch, v[0], v[1], v[2], v[3]))

            f.close()






# for k in range(10, 11):
#     bst = 0.3
#     bnt = 0.1
#     for epoch in [22]:
#         model = rcnn_model(box_score_thresh=bst, box_nms_thresh=bnt, knn_features=True, k=k)
#         model.to(device)
#         model_path = os.path.join("./models", 'model_epoch-%d-focal-adam-top%d.pt' % (epoch, k))
#         net_state_dict = torch.load(model_path)
#         model.load_state_dict(net_state_dict)
#         print("Epoch:%d\n" % epoch)
#         print_str, matrix = visualize(model, data_loader_test, device, generate_result=True, k=k)