from visualize import visualize
import os
import torch
from main import get_transform
from dataset import NBIFiveFoldDataset
from torch.utils.data import DataLoader, Subset
import utils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from model import rcnn_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 5
fold = 0

# use our dataset and defined transformations
five_fold_dataset = NBIFiveFoldDataset(get_transform(train=True))
data_loader_train = DataLoader(five_fold_dataset[fold]["train"], batch_size=1, shuffle=False, collate_fn=utils.collate_fn)
data_loader_test = DataLoader(five_fold_dataset[fold]["val"], batch_size=1, shuffle=False, collate_fn=utils.collate_fn)
data_loader_test_feature_maps = DataLoader(Subset(five_fold_dataset[fold]["val"], [0]), batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

box_score_threshs = [0.3]
box_nms_threshs = [0.3]

# for k in range(1, 11):
for k in [4]:
    for bst in box_score_threshs:
        for bnt in box_nms_threshs:
            results = {}
            f = open('./experiments/resnet_34_fold%d_top%d_bst_%.2f_bnt_%.2f.txt' % (fold, k, bst, bnt), 'w')
            for epoch in range(6, 51, 2):
            # for epoch in [32]:
                model = rcnn_model(box_score_thresh=bst, box_nms_thresh=bnt, knn_features=True, k=k)
                model.to(device)
                model_path = os.path.join("./models", 'resnet_34_fold%d_epoch-%d-top%d.pt' % (fold, epoch, k))
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


# for bst in box_score_threshs:
#     for bnt in box_nms_threshs:
#         results = {}
#         model = rcnn_model(box_score_thresh=bst, box_nms_thresh=bnt, knn_features=True, k=1)
#         model.to(device)
#         model_path = os.path.join("./models", 'fold%d_epoch-%d-top%d.pt' % (0, 32, 1))
#         net_state_dict = torch.load(model_path)
#         model.load_state_dict(net_state_dict)
#
#         print_str, matrix = visualize(model, data_loader_test_feature_maps, device, generate_result=True)

# epoch=32
# bst=0.3
# bnt=0.1
#
# model = rcnn_model(box_score_thresh=bst, box_nms_thresh=bnt, knn_features=True, k=1)
# model.to(device)
#
# model_path = os.path.join("./models", 'fold%d_epoch-%d-top%d.pt' % (0, 32, 1))
# net_state_dict = torch.load(model_path)
# model.load_state_dict(net_state_dict)
# model.eval()
#
# import numpy as np
# from density import gaussian_filter_density, kernels_dict
# from PIL import Image
# from torchvision import transforms
# save_path = "./density_maps_predict"
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
#
# for count, (images, targets) in enumerate(data_loader_test):
#
#     gt_boxes = targets[0]["boxes"].to(device)
#     gt_labels = targets[0]["labels"].to(device)
#
#     images = list(image.to(device) for image in images)
#     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#     out = model(images, targets)
#     print(out)
#     boxes = out[0]['boxes']
#     labels = out[0]['labels']
#
#     img = transforms.ToPILImage()(images[0].cpu())
#     img_array = np.array(img, dtype=np.float32)
#     img_cls = np.array(img, dtype=np.float32)
#     # img.show()
#
#     cls_points = [[], [], [], []]  # A, B1, B2, B3
#
#     for i, box in enumerate(boxes):
#         x1, y1, x2, y2 = tuple(box)
#         x1 = int(x1)
#         y1 = int(y1)
#         x2 = int(x2)
#         y2 = int(y2)
#         center_x = (x1 + x2) // 2
#         center_y = (y1 + y2) // 2
#         w = x2 - x1
#         h = y2 - y1
#         size = max(w, h)
#         cls_points[labels[i] - 1].append((center_x, center_y, size))
#
#     w = img.size[0]
#     h = img.size[1]
#     density_map = np.zeros((5, h, w), dtype=np.float32)
#     density_map[0] = 1e-10
#     dm = []
#
#     colors = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)]  # neg, A, b1-3
#     for i, points in enumerate(cls_points):
#         if len(points) > 0:
#             density_map[i + 1] = gaussian_filter_density(points, map_h=h, map_w=w, kernels_dict=kernels_dict)
#             log_den = np.log(density_map[i + 1] + 1e-12)
#             max_log = np.max(log_den)  # -> 255
#             min_log = np.max(log_den) - np.log(1000)  # -> 0
#             color = (log_den - min_log) / (max_log - min_log) * 90
#             print(color)
#             color = np.clip(color, 0, 255)
#             for c in range(3):
#                 img_array[:, :, c] += color * colors[i + 1][c]
#
#     np.save(os.path.join(save_path, "%d.npy" % count), density_map[1:])
#
#     img_array = np.clip(img_array, 0, 255)
#     final = Image.fromarray(np.uint8(img_array))
#     final.save(os.path.join(save_path, "%d.png" % count))